# EduRuta 3 – Agente predictivo de deserción escolar (prototipo)

Prototipo de Machine Learning que estima qué estudiantes tienen mayor probabilidad de abandonar el colegio y **explica por qué**, en lenguaje sencillo. Hace parte de la propuesta *EduRuta 3: Sistema Predictivo y Conexión Educativa* (Territorio INN 2026, comuna 3 Manrique, Medellín).

> **Aviso:** todos los datos son **ficticios**. Este prototipo demuestra el flujo técnico, no la precisión real de predicción. Una alerta es una invitación a acompañar al estudiante, no un diagnóstico.

## Contenido

| Archivo | Descripción |
|---|---|
| `generar_datos.py` | Genera `estudiantes.json` con estudiantes inventados |
| `estudiantes.json` | Datos de ejemplo (300 estudiantes) |
| `agente_eduruta.py` | Entrena el modelo, calcula el riesgo y explica cada alerta |
| `reporte_riesgo.json` | Salida del agente (se crea al ejecutarlo) |

## Requisitos

- Python 3.9 o superior
- Librerías: `scikit-learn` y `numpy`

```bash
pip install scikit-learn numpy
```

## Uso rápido

```bash
# 1. (Opcional) Regenerar los datos de ejemplo
python generar_datos.py              # 300 estudiantes
python generar_datos.py --n 500      # otra cantidad

# 2. Ejecutar el agente
python agente_eduruta.py                    # top 15 de estudiantes activos con más riesgo
python agente_eduruta.py --top 30           # top 30
python agente_eduruta.py --buscar "Uribe"   # ficha por id, nombre o apellido
```

## Qué datos usa

Por cada estudiante, el JSON incluye:

- Nombres y apellidos, grado, grupo y estrato (1 a 4)
- Notas finales de 5 materias (Matemáticas, Lenguaje, Ciencias Naturales, Ciencias Sociales, Inglés) en 4 periodos, escala 1.0 a 5.0
- Faltas comportamentales: si tuvo o no, y de qué tipo (I, II o III) y descripción
- `abandono_escolar`: si abandonó o no (histórico que el modelo aprende a predecir)

## Cómo funciona

1. **Variables:** a partir de las notas, el estrato y las faltas se calculan promedios por periodo, tendencia (P4 − P1), materias perdidas (promedio < 3.0), peor materia, mayor caída y número de faltas (graves = Tipo III).
2. **Modelo:** Random Forest de `scikit-learn`, con balanceo de clases porque los casos de abandono son minoría.
3. **Riesgo:** probabilidad de abandono por estudiante, calculada con validación cruzada para que ningún estudiante se evalúe con un modelo que ya lo vio. Niveles: **BAJO** (< 30 %), **MEDIO** (30–59 %), **ALTO** (≥ 60 %).
4. **Explicación:** para cada estudiante se reemplaza cada variable por un valor típico y se mide cuánto baja el riesgo. Las variables que más lo bajan se traducen a frases entendibles.
5. **Acciones sugeridas:** recomendaciones según los motivos (apoyo académico, orientación escolar, conversación con la familia, vinculación a clubes o media técnica).

## Ejemplo de salida

```
EST-0269 | Salomé Manuela Uribe Vélez | grado 9-03 | estrato 1
  Riesgo de deserción: 80%  ->  nivel ALTO
  ¿Por qué esta alerta?
    • Rendimiento académico por debajo de lo esperado (promedio 2.9); su materia más débil es Inglés (2.7)
    • Sus notas vienen cayendo durante el año (de 3.3 en P1 a 2.7 en P4)
  Acciones sugeridas:
    → Apoyo académico / nivelación en las materias débiles
    → Conversación con la familia / acudiente
```

## Limitaciones

- Los datos son sintéticos y el abandono se generó con reglas, por lo que el modelo descubre patrones que ya estaban incorporados. Las métricas no son evidencia de desempeño real.
- No incluye inasistencias, una de las señales más importantes en un caso real.
- Con datos reales hay que revisar sesgos (por ejemplo, que el estrato perjudique injustamente a ciertos estudiantes) y definir los umbrales de alerta junto con orientadores y coordinadores.
- Los datos reales de menores requieren cumplir la Ley 1581 de 2012 (protección de datos personales) y contar con las autorizaciones correspondientes.

## Infraestructura de AWS Propuesta
[Arquitectura](https://lucid.app/lucidchart/85b4add2-f705-4785-b756-8c8059d75bcc/edit?invitationId=inv_cdeaf471-f129-483c-8538-fa4410dbdc62)

## Mockup Figma
[Mockup](https://clash-chili-30530080.figma.site/)