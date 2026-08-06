import json
import os
import streamlit as st
from openai import OpenAI
from ddgs import DDGS
import numpy as np

# Groq: gratis, sin tarjeta, Key en variable de entorno, no escrita en el archivo
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"  # modelo gratis de Groq con soporte de tools

# parámetros exportados desde el notebook 
try:
    with open("parametros_proyeccion.json") as f:
        parametros = json.load(f)
except FileNotFoundError:
    st.warning(
        "⚠️ No se encontró parametros_proyeccion.json — usando valores de ejemplo, "
        "NO los resultados reales del análisis. Exportá el archivo desde el notebook."
    )
    parametros = {
        "ultimo_eq1": 150.0, "vol_eq1": 0.05,
        "ultimo_eq2": 320.0, "vol_eq2": 0.07
    }


def obtener_proyeccion(equipo, mes):
    if equipo == "Equipo1":
        ultimo = parametros["ultimo_eq1"]
        vol = parametros["vol_eq1"]
    else:
        ultimo = parametros["ultimo_eq2"]
        vol = parametros["vol_eq2"]

    dias = mes * 21
    banda = ultimo * vol * np.sqrt(dias) * 1.96
    lo = ultimo - banda
    hi = ultimo + banda
    return f"{equipo} en el mes {mes}: esperado {ultimo:.2f}, rango 95% [{lo:.2f}, {hi:.2f}]"


def buscar_contexto_mercado(consulta):
    """Búsqueda web gratis (DuckDuckGo), sin necesitar ninguna API key."""
    resultados = DDGS().text(consulta, max_results=3)
    if not resultados:
        return "No se encontraron resultados."
    texto = ""
    for r in resultados:
        texto += f"- {r['title']}: {r['body']} (fuente: {r['href']})\n"
    return texto


tools = [
    {
        "type": "function",
        "function": {
            "name": "obtener_proyeccion",
            "description": "Devuelve la proyección de costo (esperado y rango de 95%) de un equipo para un mes futuro (1 a 4).",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipo": {"type": "string", "enum": ["Equipo1", "Equipo2"]},
                    "mes": {"type": "integer", "description": "Mes hacia adelante, de 1 a 4"},
                },
                "required": ["equipo", "mes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_contexto_mercado",
            "description": "Busca en internet noticias o tendencias recientes de mercado relacionadas con materias primas de construcción, para enriquecer una proyección con contexto externo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "Qué buscar, en pocas palabras"},
                },
                "required": ["consulta"],
            },
        },
    },
]

FUNCIONES = {
    "obtener_proyeccion": obtener_proyeccion,
    "buscar_contexto_mercado": buscar_contexto_mercado,
}

SYSTEM_PROMPT = """Sos un asistente que explica los resultados de un análisis de costos de equipos de construcción.

CONTEXTO DEL ANÁLISIS:
- Se identificó que el precio de Equipo1 está explicado principalmente por la materia prima Y, y el de Equipo2 por Y y Z (regresión estadística, con significancia confirmada).
- Los modelos tienen un poder explicativo moderado (R² entre 0.21 y 0.24) — no prometas precisión que los datos no sostienen.
- La proyección usa un camino aleatorio: el valor esperado es el último precio real conocido, con un rango de confianza del 95% que se ensancha cuanto más lejos se proyecta.
- El horizonte recomendado es de 4 meses; más allá de eso, el rango de incertidumbre se vuelve más ancho que el propio valor esperado.

HERRAMIENTAS DISPONIBLES:
- obtener_proyeccion: usala siempre que te pregunten por un número de costo proyectado, para un equipo y mes específico. No inventes ni calcules estos números vos mismo — siempre llamá a la herramienta.
- buscar_contexto_mercado: es la herramienta para combinar la proyección con contexto de mercado externo, como pide el caso. Si el usuario ya pidió explícitamente contexto de mercado junto con la proyección, usala directo. Si solo pidió el número, dale la proyección primero y preguntale si quiere que sumes contexto de mercado antes de buscarlo — no lo hagas sin que lo confirme. Citá la fuente cuando la uses.

CÓMO RESPONDER:
- Por defecto, usá lenguaje simple y directo, en español, pensado para alguien sin formación técnica o estadística (por ejemplo, alguien de obra o de gerencia operativa, no un analista de datos). Evitá términos como "R²", "regresión", "coeficiente" o "significancia estadística" en las respuestas normales.
- Para explicar que la proyección no es exacta, decilo en criollo: "esta es una estimación con margen de error, no un número garantizado" — no hables de "poder explicativo" salvo que te lo pidan.
- Si te preguntan específicamente cómo funciona el modelo, por qué confiar en él, o algo técnico (ahí sí podés usar R², coeficientes, el nombre del método, etc.), respondé con ese nivel de detalle — pero solo cuando te lo pidan así.
- Si te preguntan por un mes fuera del horizonte recomendado (más de 4), aclará que la incertidumbre crece mucho y el número pierde utilidad práctica.
- No inventes cifras que no vengan de la herramienta obtener_proyeccion.
- Si no sabés algo o está fuera del alcance del análisis, decilo con honestidad en vez de inventar una respuesta.
"""

st.title("Agente de proyección de costos")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg.get("content") or "...")

pregunta = st.chat_input("Preguntaa lo que quieras saber sobre la proyección de costos y contexto de mercado...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            while True:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    tools=tools,
                )
                mensaje = response.choices[0].message

                if not mensaje.tool_calls:
                    st.write(mensaje.content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": mensaje.content}
                    )
                    break

                st.session_state.messages.append(mensaje)
                for tool_call in mensaje.tool_calls:
                    nombre = tool_call.function.name
                    argumentos = json.loads(tool_call.function.arguments)
                    resultado = FUNCIONES[nombre](**argumentos)
                    st.session_state.messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": resultado}
                    )
