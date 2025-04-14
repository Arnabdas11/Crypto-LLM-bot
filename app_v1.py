from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
import streamlit as st
from pydantic import BaseModel, Field
from typing import Literal
import requests, os


class Classifier(BaseModel):
    crypto_related: Literal["yes","no"] = Field(description="classifies if user query is crypto related or not")

class Router(BaseModel):
    type_of_query: Literal["Realtime_data","News_related","Sentiment","Technical_analysis"] = Field(description="Classifies the \
    query into these categories")

class Coin(BaseModel):
    coin: str = Field(description="This returns only the coin full name , example: if btc,bitcoin:bitcoin")

class CoinSymbol(BaseModel):
    coin_symbol: str = Field(description="This gives the ticker symbol of the crypto symbol, example: bitcoin:BTC")


class CryptoAssistant():
    def __init__(self, model, query):
        self.model = model
        self.query = query
    
    def main(self):
        self.str_parser = StrOutputParser()
        self.router_parser = PydanticOutputParser(pydantic_object=Router)
        
        with st.sidebar:
            st.markdown(f"**User query:** *{self.query}*")
        
        classifier_parser = PydanticOutputParser(pydantic_object=Classifier)
        classifier_prompt = PromptTemplate(
            template = "Given {query}, return a dictionary only having crypto_related:yes/no \n {format_instruct}",
            input_variables = ['query'],
            partial_variables = {"format_instruct":classifier_parser.get_format_instructions()}
        )
        classifier_chain = classifier_prompt | self.model | classifier_parser
        crypto_r = classifier_chain.invoke({'query':self.query}).crypto_related
        with st.sidebar:
            st.write(f"**Is this Crypto related:** *{crypto_r}*")

        decider_chain = RunnableBranch(
            (lambda x: x.crypto_related=='yes', lambda x: self.crypto_router_fn()),
            (lambda x: x.crypto_related=='no', lambda x: self.not_crypto_fn()),
            RunnableLambda(lambda x: "Not able to decide if it is crypto related or not")
        )

        final_chain = classifier_chain | decider_chain 
        final_result = final_chain.invoke({"query":self.query})
        return final_result
    
    def crypto_router_fn(self):
        crypto_prompt = PromptTemplate(
            template = "As the user query is related to crypto , given the query {query} catergorize it into type of query \n {format_instruction}",
            input_variables = ["query"],
            partial_variables={"format_instruction":self.router_parser.get_format_instructions()}
        )
        crypto_chain = crypto_prompt | self.model | self.router_parser
        query_type = crypto_chain.invoke({"query":self.query})
        with st.sidebar:
            st.markdown(f"**Query type detected:** *{query_type.type_of_query}*")

        combined_input = {
            "type_of_query": query_type.type_of_query,
            "query": self.query
        }

        router_chain = RunnableBranch(
            (lambda x: x["type_of_query"]=="Realtime_data", lambda x: self.realtime_chain_fn().invoke({"query": x["query"]})),
            (lambda x: x["type_of_query"]=="News_related", lambda x: self.news_chain_fn().invoke({"query": x["query"]})),
            (lambda x: x["type_of_query"]=="Sentiment", lambda x: self.sentiment_chain_fn().invoke({"query": x["query"]})),
            (lambda x: x["type_of_query"]=="Technical_analysis", lambda x: self.tech_chain_fn().invoke({"query": x["query"]})),
            RunnableLambda(lambda x: "Not related to any of the categories")
        )
        
        result = router_chain.invoke(combined_input)
        return result


    def not_crypto_fn(self):
        not_crypto_prompt = PromptTemplate(
            template = "As the user query is not related to crypto, give a proper feedback to the query: {query}",
            input_variables=["query"]
        )
        not_crypto_chain = not_crypto_prompt | self.model | self.str_parser
        return not_crypto_chain

    def realtime_chain_fn(self):
        with st.sidebar:
            st.markdown(f"**Entering realtime chain with query:** *{self.query}*")
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
        with st.sidebar:
            st.markdown(f"**Entering news chain with query:** *{self.query}*")
        news_fetched = self.fetch_news()
        news_context_assign = RunnablePassthrough.assign(context = lambda x: news_fetched)
        news_prompt = PromptTemplate(
            template = "Given this latest news {context}, answer the user query: {query}",
            input_variables = ["context","query"]
        )
        
        news_chain = news_context_assign | news_prompt | self.model | self.str_parser
        return news_chain

    def sentiment_chain_fn(self):
        with st.sidebar:
            st.markdown(f"**Entering sentiment chain with query:** *{self.query}*")
        sentiment_fetched = self.fetch_sentiment()
        sentiment_context_assign = RunnablePassthrough.assign(context = lambda x: sentiment_fetched)
        sentiment_prompt = PromptTemplate(
            template = "Given this sentiment {context}, answer the user query: {query}",
            input_variables = ["query"]
        )
        
        sentiment_chain = sentiment_context_assign | sentiment_prompt | self.model | self.str_parser
        return sentiment_chain

    def tech_chain_fn(self):
        with st.sidebar:
            st.markdown(f"**Entering technical chain with query:** *{self.query}*")
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
    model = ChatOpenAI(model="gpt-3.5-turbo")
    st.set_page_config(page_title="Crypto Bot - V1 (Without Agents)")
    st.header("Crypto LLM Bot - v1 (Without Agents)")
    with st.sidebar:
        st.header("**Crypto LLM Bot - V1 info**")
        st.write("="*50)
        st.markdown("- **Realtime Chain**: Realtime data - Price, 24h Volume")
        st.markdown("- **News Chain**: Latest News of the market")
        st.markdown("- **Sentiment Chain**: Current Sentiment of the market")
        st.markdown("- **Technical Chain**: Technical indicators - RSI, EMA, MACD")
        st.header("**Decision/Execution Flow**")
        st.write("="*50)
    
    query = st.text_input("Write your query here:")
    ask_btn = st.button("Ask Bot")
    if query and ask_btn:
        cryptoai = CryptoAssistant(model, query)
        final_result = cryptoai.main()
        st.write(f"{final_result}")



