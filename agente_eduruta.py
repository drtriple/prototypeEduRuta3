"""
EduRuta 3 - Agente predictivo de riesgo de deserción escolar (prototipo)
------------------------------------------------------------------------
1. Lee estudiantes.json
2. Convierte notas, estrato y faltas en variables (features)
3. Entrena un Random Forest con el histórico (columna 'abandono_escolar')
4. Calcula la probabilidad de riesgo de cada estudiante
5. Explica EN LENGUAJE SENCILLO por qué cada estudiante está en riesgo

Requisitos: pip install scikit-learn numpy
Uso:
    python agente_eduruta.py # ranking de los 15 con más riesgo
    python agente_eduruta.py --top 30
    python agente_eduruta.py --buscar "Zapata" # ficha de un estudiante por apellido/nombre/id

IMPORTANTE: es un prototipo con datos ficticios. Una alerta es una invitación a acompañar al estudiante, no un diagnóstico ni una sentencia.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

MATERIAS = ["Matemáticas", "Lenguaje", "Ciencias Naturales", "Ciencias Sociales", "Inglés"]
PERIODOS = ["P1", "P2", "P3", "P4"]
NOTA_MINIMA = 3.0 # nota mínima aprobatoria

FEATURES = [
    "promedio_general",
    "promedio_p1",
    "promedio_p4",
    "tendencia", # promedio P4 - promedio P1
    "materias_perdidas", # materias con promedio anual < 3.0
    "peor_materia", # menor promedio anual entre materias
    "mayor_caida", # mayor caída P1 -> P4 en una materia
    "estrato",
    "num_faltas",
    "faltas_graves", # Tipo III
]


# ---------------------------------------------------------------- datos
def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)

def extraer_features(e):
    notas = e["notas_finales"]
    prom_periodo = [np.mean([notas[m][p] for m in MATERIAS]) for p in PERIODOS]
    prom_materia = {m: np.mean([notas[m][p] for p in PERIODOS]) for m in MATERIAS}
    caidas = [notas[m]["P1"] - notas[m]["P4"] for m in MATERIAS]
    faltas = e["faltas_comportamentales"]["detalle"]

    return {
        "promedio_general": float(np.mean(prom_periodo)),
        "promedio_p1": float(prom_periodo[0]),
        "promedio_p4": float(prom_periodo[3]),
        "tendencia": float(prom_periodo[3] - prom_periodo[0]),
        "materias_perdidas": sum(v < NOTA_MINIMA for v in prom_materia.values()),
        "peor_materia": float(min(prom_materia.values())),
        "mayor_caida": float(max(caidas)),
        "estrato": e["estrato"],
        "num_faltas": len(faltas),
        "faltas_graves": sum(f["tipo"] == "Tipo III" for f in faltas),
    }

def matriz(estudiantes):
    filas = [extraer_features(e) for e in estudiantes]
    return np.array([[f[c] for c in FEATURES] for f in filas], dtype=float), filas


# ------------------------------------------------------------ explicación
def redactar_motivo(feature, f, e):
    """Convierte una variable en una frase entendible para un coordinador."""
    notas = e["notas_finales"]
    if feature in ("promedio_general", "promedio_p4", "peor_materia"):
        prom_materia = {m: np.mean([notas[m][p] for p in PERIODOS]) for m in MATERIAS}
        peor = min(prom_materia, key=prom_materia.get)
        return (f"Rendimiento académico por debajo de lo esperado (promedio {f['promedio_general']:.1f}); su materia más débil es {peor} ({prom_materia[peor]:.1f})")
    
    if feature in ("tendencia", "promedio_p1"):
        return (f"Sus notas vienen cayendo durante el año (de {f['promedio_p1']:.1f} en P1 a {f['promedio_p4']:.1f} en P4)")
    
    if feature == "mayor_caida":
        m = max(MATERIAS, key=lambda x: notas[x]["P1"] - notas[x]["P4"])
        return f"Caída fuerte en {m} (de {notas[m]['P1']:.1f} a {notas[m]['P4']:.1f})"
    
    if feature == "materias_perdidas":
            return f"Va perdiendo {f['materias_perdidas']} materia(s) con promedio anual inferior a 3.0"
        
    if feature == "num_faltas":
        return f"Registra {f['num_faltas']} falta(s) comportamental(es)"
    
    if feature == "faltas_graves":
        return f"Tiene {f['faltas_graves']} falta(s) grave(s) (Tipo III)"
    
    if feature == "estrato":
        return f"Contexto socioeconómico de estrato {f['estrato']}, que se asocia a mayor vulnerabilidad"
    
    return feature


def explicar(modelo, X, idx, medianas, feats_estudiante, estudiante):
    """
    Explicación local sencilla: para cada variable se reemplaza el valor del
    estudiante por el valor 'típico' (mediana) y se mide cuánto baja el riesgo.
    Las variables que más lo bajan son las que más empujan su riesgo hacia arriba.
    """
    base = modelo.predict_proba(X[idx:idx + 1])[0, 1]
    aportes = []
    for j, nombre in enumerate(FEATURES):
        x = X[idx:idx + 1].copy()
        x[0, j] = medianas[j]
        aportes.append((nombre, base - modelo.predict_proba(x)[0, 1]))
    aportes.sort(key=lambda t: t[1], reverse=True)
    
    # Se evitan motivos repetidos (varias variables pueden describir lo mismo)
    vistos, resultado = set(), []
    for n, a in aportes:
        if a <= 0.01:
            break
        texto = redactar_motivo(n, feats_estudiante, estudiante)
        if texto not in vistos:
            vistos.add(texto)
            resultado.append((n, a))
        if len(resultado) == 3:
            break
    return resultado

def nivel(p):
    if p >= 0.60:
        return "ALTO"
    if p >= 0.30:
        return "MEDIO"
    return "BAJO"

def acciones_sugeridas(feats_clave, f):
    acciones = []
    claves = {n for n, _ in feats_clave}
    if claves & {"promedio_general", "promedio_p4", "peor_materia", "mayor_caida", "materias_perdidas", "tendencia", "promedio_p1"}:
        acciones.append("Apoyo académico / nivelación en las materias débiles")
    if claves & {"num_faltas", "faltas_graves"}:
        acciones.append("Remisión a orientación escolar y seguimiento de convivencia")
    if "estrato" in claves:
        acciones.append("Revisar programas de apoyo (alimentación, transporte, subsidios)")
    acciones.append("Conversación con la familia / acudiente")
    acciones.append("Vincular a un club o a la media técnica para fortalecer el arraigo")
    return acciones[:4]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="estudiantes.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--buscar", default=None, help="id, nombre o apellido a consultar")
    args = ap.parse_args()

    estudiantes = cargar(args.datos)
    X, feats = matriz(estudiantes)
    y = np.array([int(e["abandono_escolar"]) for e in estudiantes])

    # 1) Validación honesta del modelo (datos que no vio al entrenar)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    prueba = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
    min_samples_leaf=3, random_state=7).fit(X_tr, y_tr)
    prob_te = prueba.predict_proba(X_te)[:, 1]
    print("=" * 70)
    print("EDURUTA 3 - Agente predictivo de deserción (prototipo, datos ficticios)")
    print("=" * 70)
    print(f"Estudiantes analizados: {len(estudiantes)} | Abandono histórico: {y.mean():.1%}")
    print(f"Calidad del modelo (AUC en datos de prueba): {roc_auc_score(y_te, prob_te):.2f}")
    print("(umbral de alerta usado en esta prueba: 35 %)")
    print(classification_report(y_te, (prob_te >= 0.35).astype(int),target_names=["Continúa", "Abandona"], zero_division=0))

    # 2) Modelo final + importancia global de variables
    modelo = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",min_samples_leaf=3, random_state=7).fit(X, y)
    print("Variables que más pesan en el modelo:")
    
    for n, imp in sorted(zip(FEATURES, modelo.feature_importances_), key=lambda t: -t[1])[:5]:
        print(f" - {n:<18} {imp:.1%}")

    # 3) Riesgo de cada estudiante (validación cruzada: ninguno se evalúa con un modelo que lo haya visto durante el entrenamiento)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    riesgo = cross_val_predict(
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",min_samples_leaf=3, random_state=7),X, y, cv=cv, method="predict_proba")[:, 1]

    medianas = np.median(X, axis=0)
    orden = np.argsort(-riesgo)

    # 4) Resultados
    if args.buscar:
        q = args.buscar.lower()
        seleccion = [i for i in orden if q in f"{estudiantes[i]['id']} {estudiantes[i]['nombres']} {estudiantes[i]['apellidos']}".lower()][:5]
        titulo = f"Resultados de la búsqueda '{args.buscar}'"
    else:
        seleccion = [i for i in orden if not estudiantes[i]["abandono_escolar"]][:args.top]
        titulo = f"Los {args.top} estudiantes ACTIVOS con mayor riesgo"

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    resultados = []
    
    for i in seleccion:
        e, f = estudiantes[i], feats[i]
        motivos = explicar(modelo, X, i, medianas, f, e)
        print(f"\n{e['id']} | {e['nombres']} {e['apellidos']} | grado {e['grado']}-{e['grupo']} | estrato {e['estrato']}")
        print(f" Riesgo de deserción: {riesgo[i]:.0%} -> nivel {nivel(riesgo[i])}")
        print(" ¿Por qué esta alerta?")
        
        for n, _ in motivos:
            print(f" • {redactar_motivo(n, f, e)}")
            
        if e["faltas_comportamentales"]["detalle"]:
            det = ", ".join(x["falta"] for x in e["faltas_comportamentales"]["detalle"])
            print(f" Faltas registradas: {det}")
            
        print(" Acciones sugeridas:")
        for a in acciones_sugeridas(motivos, f):
            print(f" → {a}")
            
        resultados.append({
            "id": e["id"],
            "estudiante": f"{e['nombres']} {e['apellidos']}",
            "grado": e["grado"], "grupo": e["grupo"],
            "probabilidad_riesgo": round(float(riesgo[i]), 3),
            "nivel": nivel(riesgo[i]),
            "motivos": [redactar_motivo(n, f, e) for n, _ in motivos],
        })

    # 5) Resumen por grado (lo que vería un coordinador) y archivo de salida
    print("\n" + "-" * 70)
    print("Riesgo promedio por grado (todos los estudiantes):")
    for g in sorted({e["grado"] for e in estudiantes}):
        ids = [i for i, e in enumerate(estudiantes) if e["grado"] == g]
        altos = sum(riesgo[i] >= 0.60 for i in ids)
        print(f" Grado {g:>2}: riesgo promedio {riesgo[ids].mean():.0%} | {altos} en riesgo alto")

    with open("reporte_riesgo.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)
    print("\nReporte guardado en 'reporte_riesgo.json'")
    print("Recuerda: una alerta es una invitación a acompañar, no un diagnóstico.")


if __name__ == "__main__":
    main()