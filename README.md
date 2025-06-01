# Buddy

1. Buddy es un agente de IA implementado con la librería LangGraph de python
2. Usa esta clave de API para Google AI Studio: AIzaSyAAfE6ydHeGx9-VVVVMbBLcMrB8QtGdpfE
3. Implementa la arquitectura plan-and-execute explicada en este documento: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/#create-the-graph
4. Usa el modelo gemini-2.5-flash-preview-04-17 para la planificacion de tareas
5. Usa ChatGoogleGenerativeAI para implementar la planificacion de tareas
6. Usa el modelo gemini-2.5-flash-preview-05-20 para la ejecucion de tareas
7. El llm encargado de ejecutar las tareas usa la Tool de langgraph llamada ShellTool, este es un ejemplo de como añadir esa ShellTool a un agente:

```python
from langchain_community.tools import ShellTool

agent = create_react_agent(
    model=get_llm(config),
    tools=[ShellTool()],
    prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
```
8. El agente ejecutor es una agente ReAct implementado con la funcion create_react_agent
9. La salida al usuario debe estar formateada con rich.Markdown para mejorar la experiencia del usuario
10. El codigo debe ser limpio, compacto y legible