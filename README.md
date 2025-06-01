# Buddy

- Buddy es un agente de IA escrito en python usando las librerias de LangGraph.
- Usa una arquitectura plan-and-execute como la descrita en esta url: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
- Usa AI Studio de Google con la libreria nativa de LangGraph para llamar al modelo gemini-2.5-flash-preview-04-17
- Tiene un parametro --prompt con el que recibe las instrucciones del usuario, este parametro puede ser una cadena o un fichero del que lee las instrucciones.
- Tiene un parametro --context con el que carga en el contexto del agente informacion que le pasa el usuario, puede ser un fichero del que leer su contenido o un directorio del que ha de leer recursivamente todos los ficheros que contiene
- No tiene capacidad para hacer busquedas en internet con tavily_search
- Muestra al usuario el plan que va a ejecutar y como ejecuta cada paso 
