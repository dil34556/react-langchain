import os 
import json
from django.http import JsonResponse
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain import LLMMathChain
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool, DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


@csrf_exempt
@require_POST
def chatbot(request):
    try:
        data = json.loads(request.body.decode('utf-8'))
        user_message = data.get('message', '').lower()
        api_key = data.get('api_key', '')  # Extract API key from request payload
        bot_type = data.get('bot_type', '')  # Extract bot type from request payload

        if bot_type == 'huggingFace':
            # Existing Hugging Face bot logic
            llm = HuggingFaceHub(huggingfacehub_api_token=api_key, repo_id='tiiuae/falcon-7b-instruct', model_kwargs={'temperature': 0.7, "max_new_token": 500})
            template = """Questions: {questions}
            Answer: let's give a detailed answer.
            """
            prompt = PromptTemplate(template=template, input_variables=["questions"])
            chain = LLMChain(prompt=prompt, llm=llm)
            user_message = user_message.replace('qestions', 'questions')
            bot_response = chain.run(user_message)

        elif bot_type == 'mathBot':
            llm=OpenAI(temperature=0,openai_api_key=api_key)
            llm_math_chain = LLMMathChain(llm=llm, verbose=True)
            # api = "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"
            llm = OpenAI(temperature=0, openai_api_key=api_key)
            search = DuckDuckGoSearchRun()
            llm_math_chain = LLMMathChain(llm=llm, verbose=True)

            tools = [
                Tool(
                    name="Search",
                    func=search.run,
                    description="You are a chatbot that can answer questions about specific events",
                ),
                Tool(
                    name="Math",
                    func=llm_math_chain.run,
                    description="You are a chatbot that can answer questions related to math",
                )
            ]
            prefix = """Interact with a human by answering the questions by accessing the following tools:"""
            suffix = """Begin!"

            {history}
            Query: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "history", "agent_scratchpad"],
            )
            memory = ConversationBufferMemory(memory_key="history")
            llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=memory
            )
            
            
            bot_response = agent_chain.invoke(user_message)

            
            bot_response = bot_response['output']

        else:
            bot_response = 'Unknown bot type or default response.'

        return JsonResponse({'message': bot_response})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)