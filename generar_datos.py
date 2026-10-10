"""
EduRuta 3 - Generador de datos sintéticos (ficticios)
-----------------------------------------------------
Crea el archivo estudiantes.json con estudiantes inventados.
NINGÚN dato es real. Sirve únicamente para probar el modelo de Machine Learning.

Uso:
python generar_datos.py # 300 estudiantes
python generar_datos.py --n 500 # otra cantidad
"""
import argparse
import json
import math
import random

SEMILLA = 42

NOMBRES_M = ["Juan", "Santiago", "Mateo", "Samuel", "Sebastián", "Andrés", "Daniel", "Julián", "Camilo", "Esteban", "Jhon", "Brayan", "Kevin", "Alejandro", "David", "Nicolás",]
NOMBRES_F = ["Valentina", "Sofía", "Isabella", "Mariana", "Luisa", "Daniela", "Camila", "Laura","Juliana", "Karen", "Manuela", "Salomé", "Yuliana", "Paula", "Natalia", "Gabriela",]
APELLIDOS = [
    "Gómez", "Rodríguez", "Martínez", "López", "García", "Hernández", "Ramírez", "Torres", "Zapata", "Restrepo", "Ospina", "Cardona", "Mejía", "Vélez", "Giraldo", "Montoya",
    "Henao", "Arango", "Quintero", "Uribe", "Betancur", "Agudelo", "Londoño", "Álvarez", "Muñoz", "Salazar", "Duque", "Marín", "Ríos", "Cano", "Jaramillo", "Tobón",
]
MATERIAS = ["Matemáticas", "Lenguaje", "Ciencias Naturales", "Ciencias Sociales", "Inglés"]
PERIODOS = ["P1", "P2", "P3", "P4"]

# (tipo, descripción) - inspirado en la clasificación habitual de manuales de convivencia
FALTAS = {
    "Tipo I": [
        "Llegadas tarde reiteradas",
        "Uso del celular en clase",
        "Incumplimiento del uniforme",
        "Interrupciones constantes en clase",
    ],
    "Tipo II": [
        "Agresión verbal a un compañero",
        "Fuga o evasión de clases",
        "Daño a material del colegio",
        "Matoneo o intimidación",
    ],
    "Tipo III": [
        "Agresión física",
        "Porte o consumo de sustancias psicoactivas",
        "Hurto",
        "Amenazas graves",
    ],
}

def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def limitar(x, minimo=1.0, maximo=5.0):
    return max(minimo, min(maximo, x))

def generar_estudiante(i):
    # Riesgo "oculto" del estudiante (no se guarda): mueve notas, faltas y abandono
    riesgo = random.betavariate(1.2, 3.0)

    grado = random.choice([6, 7, 8, 9, 10, 11])
    grupo = f"{random.randint(1, 3):02d}"
    estrato = random.choices([1, 2, 3, 4], weights=[0.30, 0.45, 0.22, 0.03])[0]

    # Estratos bajos tienen algo más de riesgo en los datos simulados
    riesgo = min(1.0, riesgo + {1: 0.08, 2: 0.03, 3: 0.0, 4: -0.03}[estrato])

    # Notas finales por materia y periodo (escala colombiana 1.0 a 5.0)
    base = 4.0 - 1.1 * riesgo + random.gauss(0, 0.30)
    notas = {}
    for materia in MATERIAS:
        sesgo_materia = random.gauss(0, 0.25)
        pendiente = -0.30 * riesgo + random.gauss(0, 0.06) # caída por periodo
        notas[materia] = {}
        for k, periodo in enumerate(PERIODOS):
            valor = base + sesgo_materia + pendiente * k + random.gauss(0, 0.22)
            notas[materia][periodo] = round(limitar(valor), 1)

    # Faltas comportamentales
    detalle = []
    if random.random() < 0.05 + 0.65 * riesgo:
        cantidad = 1 + int(random.random() < 0.5 + riesgo) + int(random.random() < riesgo)
        for _ in range(cantidad):
            tipo = random.choices(
                ["Tipo I", "Tipo II", "Tipo III"],
                weights=[0.55 - 0.3 * riesgo, 0.35, 0.10 + 0.3 * riesgo],
            )[0]
            detalle.append({"tipo": tipo, "falta": random.choice(FALTAS[tipo])})

    graves = sum(1 for f in detalle if f["tipo"] == "Tipo III")

    # Etiqueta histórica: ¿abandonó o desertó? (simulada a partir del riesgo oculto)
    z = -3.6 + 5.0 * riesgo + 0.35 * graves + (0.25 if estrato == 1 else 0.0)
    abandono = random.random() < sigmoide(z)

    lista_nombres = random.choice([NOMBRES_M, NOMBRES_F])

    return {
        "id": f"EST-{i:04d}",
        "nombres": " ".join(random.sample(lista_nombres, 2)),
        "apellidos": " ".join(random.sample(APELLIDOS, 2)),
        "grado": grado,
        "grupo": grupo,
        "estrato": estrato,
        "notas_finales": notas,
        "faltas_comportamentales": {
        "tuvo_faltas": len(detalle) > 0,
        "detalle": detalle,
        },
        "abandono_escolar": abandono,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="cantidad de estudiantes")
    parser.add_argument("--salida", default="estudiantes.json")
    args = parser.parse_args()

    random.seed(SEMILLA)
    estudiantes = [generar_estudiante(i) for i in range(1, args.n + 1)]

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(estudiantes, f, ensure_ascii=False, indent=2)

    desertores = sum(e["abandono_escolar"] for e in estudiantes)
    print(f"Archivo '{args.salida}' creado con {len(estudiantes)} estudiantes ficticios.")
    print(f"Abandono simulado: {desertores} ({desertores / len(estudiantes):.1%})")


if __name__ == "__main__":
    main()