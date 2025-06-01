# Buddy

1. Buddy es un agente de IA implementado con la librería LangGraph de python
2. El objetivo de Buddy es ayudar al usuario en tareas de administracion de sistemas y de desarrollo de software
3. Usa esta clave de API para Google AI Studio: AIzaSyAAfE6ydHeGx9-VVVVMbBLcMrB8QtGdpfE
4. Implementa la arquitectura plan-and-execute explicada en este documento: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/#create-the-graph
5. Usa el modelo gemini-2.5-flash-preview-04-17 para la planificacion de tareas
6. Usa ChatGoogleGenerativeAI para implementar la planificacion de tareas
7. Usa el modelo gemini-2.5-flash-preview-05-20 para la ejecucion de tareas
8. El llm encargado de ejecutar las tareas usa la Tool de langgraph llamada ShellTool, este es un ejemplo de como añadir esa ShellTool a un agente:

```python
from langchain_community.tools import ShellTool

agent = create_react_agent(
    model=get_llm(config),
    tools=[ShellTool()],
    prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
```
9. El agente ejecutor es una agente ReAct implementado con la funcion create_react_agent
10. La salida al usuario debe estar formateada con rich.Markdown para mejorar la experiencia del usuario
11. El codigo debe ser limpio, compacto y legible
12. La aplicacion tiene un parametro --prompt que son las instrucciones del usuario para el agente, este parametro puede ser una string o un fichero del que lee las instrucciones
13. La aplicacion tiene un paremetro --context que es informacion que se carga en el contexto para poder realizar su trabajo, pude ser un fichero del que lee su contenido o un directorio que del que lee recursivamente todos los ficheros de texto
14. Ejemplos de llamadas que puede resolver el agente:

```bash
buddy --context . --prompt "analiza el proyecto y añade una opcion para listar ficheros, crea una bateria de test end-to-end y ejecutala para ver si funciona, si no funciona arregla los fallos y vuelve a intentarlo"
buddy --context file.py --prompt "mejora la sintaxis del fichero, al terminar ejecutalo para ver si funciona"
buddy --prompt "crea una app en golang, que imprima el contenido de la url https://google.com, compilalo y asegurate de que se ejecuta correctamente"
```

## Code Structure

The codebase was recently refactored to improve modularity and resolve import issues. Here's a brief overview:

-   **Core Agent Logic**: The main LangGraph definition, state management, and core functions like LLM/agent runnable setup are located in `buddy_ai/agent.py`.
-   **Graph Nodes**: Individual nodes that form the LangGraph (e.g., planner, executor, replanner, human approval, deciders) have been separated into their own modules within the `buddy_ai/nodos/` directory. For example:
    -   `buddy_ai/nodos/planner.py`
    -   `buddy_ai/nodos/executor.py`
    -   `buddy_ai/nodos/replanner.py`
    -   `buddy_ai/nodos/human_approval.py`
    -   `buddy_ai/nodos/deciders.py`
-   **Shared Instances**: To manage objects that need to be globally accessible by different parts of the agent (like configured LLM instances or the Rich console object) and to prevent circular dependencies, a dedicated module `buddy_ai/shared_instances.py` is used. These objects are initialized by `agent.py` (or `__main__.py`) and stored in `shared_instances.py`, from where nodes can import them directly.
-   **Command-Line Interface**: The entry point for the CLI (`python -m buddy_ai ...`) is now `buddy_ai/__main__.py`. This file handles argument parsing, configuration loading, and invoking the main agent workflow. The previous `buddy_ai/cli.py` has been removed.