from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
import streamlit as st
from pydantic import BaseModel, Field
from typing import Literal
import requests, os
from langchain.agents import tool, create_openai_functions_agent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.callbacks.base import BaseCallbackHandler


class Coin(BaseModel):
    coin: str = Field(description="This returns only the coin full name , example: if btc,bitcoin:bitcoin")

class CoinSymbol(BaseModel):
    coin_symbol: str = Field(description="This gives the ticker symbol of the crypto symbol, example: bitcoin:BTC")

class handling_callbacks(BaseCallbackHandler):
    
    def on_text(self, text):
        # print(f"Thinking: {text}")
        with st.sidebar:
            st.markdown(f"Thinking: {text}")
    
    def on_tool_start(self, tool, **kwargs):
        # print(f"Tool: `{tool.name}` Query:{query}")
        with st.sidebar:
            st.markdown(f"Tool: `{tool.name}` Query:{query}")
    
class ConsoleScratchpadPrinter(BaseCallbackHandler):
    def on_tool_start(self, tool, input: str, **kwargs) -> None:
        print(f"\n🔧 Tool: {tool.name} — Input: {input}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"✅ Tool Output: {output}")

    def on_text(self, text: str, **kwargs) -> None:
        print(f"🧠 Agent says: {text}")


class CryptoAssistant():
    def __init__(self, model, query):
        self.model = model
        self.query = query
    
    def main(self):
        self.str_parser = StrOutputParser()
        
        with st.sidebar:
            st.markdown(f"**User query:** *{self.query}*")
        
    
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful cryptocurrency assistant. Perform these steps when user asks any query :
            - Check if the user query is related to crypto or not, if not politely refuse as you are only a cryptocurrency expert.
            - If the user query is related to crypto, identify what type of query it is and call that particular tool.
            If you are not able to decide which tool to use then check the internet and reply, but mention that you have found from internet.
            - Once you have received data from the tools answer the user query

        """),
        ("user", "{query}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        
        tools = [
            tool(self.realtime_tool),
            tool(self.news_tool),
            tool(self.sentiment_tool),
            tool(self.technical_tool)
        ]
        agent = create_openai_functions_agent(
            llm=self.model,
            tools=tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
        )

        return agent_executor
    
    
    def realtime_tool(self, query):
        '''Answers queries related to realtime data like price, volume for the crypto coin'''
        return self.realtime_chain_fn().invoke({'query':query})
    
    
    def news_tool(self, query):
        '''Answers queries related to latest news about crypto market'''
        return self.news_chain_fn().invoke({'query':query})
    
    
    def sentiment_tool(self, query):
        '''Answers queries related to crypto sentiment'''
        return self.sentiment_chain_fn().invoke({'query':query})
    
    
    def technical_tool(self, query):
        '''Answers queries related to techinal indicators like rsi, ema & macd'''
        return self.tech_chain_fn().invoke({'query':query})

    def realtime_chain_fn(self):
        coin_parser = PydanticOutputParser(pydantic_object=Coin)
        get_coin_prompt = PromptTemplate(
            template = "Identify the coin in the query {query} \n {format_instruct}",
            input_variables = ["query"],
            partial_variables = {"format_instruct":coin_parser.get_format_instructions()}
            )
        coin_chain = get_coin_prompt | self.model | coin_parser
        result = coin_chain.invoke({"query":self.query})
        realtime_data_fetched = self.fetch_realtime_data(result.coin)
        context_assign = RunnablePassthrough.assign(context = lambda x: realtime_data_fetched)

        realtime_prompt =  PromptTemplate(
            template = "Given {context}, reply as per user query {query} \n",
            input_variables = ["context","query"]
        )
        realtime_chain = context_assign | realtime_prompt | self.model | self.str_parser
        return realtime_chain
    
    def news_chain_fn(self):
        news_fetched = self.fetch_news()
        news_context_assign = RunnablePassthrough.assign(context = lambda x: news_fetched)
        news_prompt = PromptTemplate(
            template = "Given this latest news {context}, answer the user query: {query}",
            input_variables = ["context","query"]
        )
        
        news_chain = news_context_assign | news_prompt | self.model | self.str_parser
        return news_chain

    def sentiment_chain_fn(self):
        sentiment_fetched = self.fetch_sentiment()
        sentiment_context_assign = RunnablePassthrough.assign(context = lambda x: sentiment_fetched)
        sentiment_prompt = PromptTemplate(
            template = "Given this sentiment {context}, answer the user query: {query}",
            input_variables = ["query"]
        )
        
        sentiment_chain = sentiment_context_assign | sentiment_prompt | self.model | self.str_parser
        return sentiment_chain

    def tech_chain_fn(self):
        coin_symbol_parser = PydanticOutputParser(pydantic_object=CoinSymbol)
        get_coin_symbol_prompt = PromptTemplate(
            template = "Identify the crypto coin symbol in the query {query} \n {format_instruct}",
            input_variables = ["query"],
            partial_variables = {"format_instruct":coin_symbol_parser.get_format_instructions()}
            )
        coin_chain = get_coin_symbol_prompt | self.model | coin_symbol_parser
        result = coin_chain.invoke({"query":self.query})
        tech_fetched = self.fetch_technical_analysis(result.coin_symbol.upper())
        tech_context_assign = RunnablePassthrough.assign(context = lambda x: tech_fetched)
        tech_prompt = PromptTemplate(
            template = "Given this technical analysis data {context}, answer the user query: {query}",
            input_variables = ["query"]
        )

        tech_chain = tech_context_assign | tech_prompt | self.model | self.str_parser
        return tech_chain
    
    def fetch_realtime_data(self,coin):
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coin}"
        response = requests.get(url)
        return response.text

    def fetch_news(self):
        response = requests.get('https://data-api.coindesk.com/news/v1/article/list',
            params={"lang":"EN","limit":10,"api_key":os.environ.get("COINDESK_API_KEY")},
            headers={"Content-type":"application/json; charset=UTF-8"}
        )
        json_response = response.json()
        total_response =''
        for i in json_response["Data"]:
            total_response+= i['BODY'] +'\n'
        return total_response

    def fetch_sentiment(self):
        url = "https://api.tokenmetrics.com/v2/sentiments?limit=1000&page=0"
        headers = {
            "accept": "application/json",
            "api_key": os.environ.get("CRYPTO_API_KEY")
        }
        response = requests.get(url, headers=headers)
        return response.json()["data"][0]
    
    def fetch_technical_analysis(self, coin_symbol):
        url = "https://api.taapi.io/bulk"
        coin_symbol_list = ["BTC","ETH","XRP","LTC","XMR"]
        if coin_symbol not in coin_symbol_list:
            return "No info for this crypto coin found"
        else:
            payload = {
                "secret": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjdmOTc1OWM4MDZmZjE2NTFlNGM5YzJiIiwiaWF0IjoxNzQ0NDAyNTYwLCJleHAiOjMzMjQ4ODY2NTYwfQ.79ecFpLjuufvSrSIFfyR6m0FjYXWIoEAbGlTCXrGaOQ",
                "construct": {
                    "exchange": "binance",
                    "symbol": f"{coin_symbol}/USDT",
                    "interval": "1d",
                    "indicators": [
                        {
                            "indicator": "rsi"
                        }, 
                        {
                            "indicator": "ema",
                            "period": 20
                        },
                        {
                            "indicator": "ema",
                            "period": 50
                        },
                        {
                            "indicator": "ema",
                            "period": 200
                        },
                        {
                            "indicator": "macd"
                        }
                    ]
                }
            }
            headers = {"Content-Type": "application/json"}
            response = requests.request("POST", url, json=payload, headers=headers)
            return response.json()['data'] 

if __name__=="__main__":
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o-mini")
    st.set_page_config(page_title="Crypto Bot - v2 (With Agent & Tool calling)")
    st.header("Crypto LLM Bot - v2 (With Agent & Tool calling)")
    with st.sidebar:
        st.header("**Crypto LLM Bot - v2 info**")
        st.write("="*50)
        st.markdown("- **Realtime Tool**: Realtime data - Price, 24h Volume")
        st.markdown("- **News Tool**: Latest News of the market")
        st.markdown("- **Sentiment Tool**: Current Sentiment of the market")
        st.markdown("- **Technical Tool**: Technical indicators - RSI, EMA, MACD")
        st.header("**Decision/Execution Flow**")
        st.write("="*50)

    
    query = st.text_input("Write your query here:")
    ask_btn = st.button("Ask Bot")
    if query and ask_btn:
        cryptoai = CryptoAssistant(model, query)
        agent = cryptoai.main()
        final_result = agent.invoke({"query":query})
        for action, observation in final_result['intermediate_steps']:
            with st.sidebar:
                st.markdown(f"**Tool to use:** *{action.tool}*")
        st.write(f"{final_result['output']}")



