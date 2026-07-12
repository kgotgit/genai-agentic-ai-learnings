memory_list = []
from langchain.messages import HumanMessage, AIMessage, SystemMessage
system_msg = SystemMessage("You are a helpful assistant.")
memory_list.append(system_msg)

user_input= input("enter your query: ")
while user_input != "exit":
    human_msg = HumanMessage(content=user_input)
    memory_list.append(human_msg)
    output = llm.invoke(memory_list)
    print(output.content)
    ai_msg = AIMessage(content=output.content)
    memory_list.append(ai_msg)
    user_input= input("enter your query: ")


Use above python code as reference and use streamlit and build an UI chat bot interface 

Create a new example