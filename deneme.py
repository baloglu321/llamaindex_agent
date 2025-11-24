import time
import os
import requests
import asyncio 
from llama_index.llms.litellm import LiteLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent,ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole


import requests

CLOUDFLARE_TUNNEL_URL = "https://retrieved-offices-myspace-cooperative.trycloudflare.com" 
OLLAMA_MODEL_ID = "gemma3:27b"
WEATHER_API = "28df0827f992105fb2d12d6c224a9509"
SERPER_API_KEY = "d713b469bec107810b2ec381f4454743bf409489"

def get_question():
    API_URL = "https://agents-course-unit4-scoring.hf.space/random-question"
    response = requests.get(API_URL).json()

    question = response.get("question")
    return question, response

class CustomError(Exception):
    pass

def WeatherInfoTool(location:str)->str:
        weather_start=time.time()
        url = f"https://api.weatherstack.com/current?access_key={WEATHER_API}"
        querystring = {"query": location}
        response = requests.get(url, params=querystring)
        data = response.json()
        city = data["location"]["name"]
        country = data["location"]["country"]
        temperature = data["current"]["temperature"]
        weather_description = data["current"]["weather_descriptions"][0]
        weather_stop=time.time()
        weather_time=weather_stop-weather_start
        print(f"⏱️ Çalışma Süresi (weather tool cevap süresi): {weather_time:.2f} saniye")
        return f"Weather in {location}: {weather_description}, {str(temperature)}°C"


weather_tool = FunctionTool.from_defaults(
    WeatherInfoTool,
    name="my_weather_tool",
    description="Fetches weather information for a given location.",
)

tool_list = [weather_tool]
llm_start_time=time.time()
print(f"LLM baglantisi kuruluyor: {OLLAMA_MODEL_ID} @ {CLOUDFLARE_TUNNEL_URL}")
llm = Ollama(
    model=OLLAMA_MODEL_ID, 
    base_url=CLOUDFLARE_TUNNEL_URL, 
    context_window=8192,
    # verbose=True ekleyerek LiteLLM'in HTTP isteklerini gorebilirsiniz.
    request_timeout="3000"
)
llm_stop_time=time.time()
llm_time=llm_stop_time-llm_start_time
print(f"⏱️ Çalışma Süresi (LLM bağlantı süresi): {llm_time:.2f} saniye")
print("LlamaIndex FunctionAgent olusturuluyor...")
""
agent = ReActAgent( 
    tools=tool_list,
    llm=llm,
    verbose=True, 
    system_prompt="Sen, kullanıcının sorularını yanıtlamak için araçları kullanabilen yetenekli bir asistansın. Hava durumu soruları için 'my_weather_tool' aracını kullan.",
)

print("\n✅ LlamaIndex Agent kurulumu tamamlandı. Test ediliyor.")
print("-" * 50)



# --- 4. ASENKRON ÇALIŞTIRMA FONKSİYONU ---
async def main():
    test_1_start=time.time()
    # TEST 1: Hava Durumu Sorgusu (Araç Kullanılmalı)
    print("## 🛠️ Test 1: Hava Durumu Sorgusu (my_weather_tool kullanılmalı) ##")
    user_query = "Şu anda Konya/Türkiyede hava nasıl?"

    # DÜZELTME: .run() yerine .achat() kullanıldı.
    response_tool = await agent.run(user_query)


    print(f"\n[AGENT YANITI]: {response_tool.response}")
    print("-" * 50)
    test_1_stop=time.time()
    test_1_time=test_1_stop-test_1_start
    print(f"⏱️ Çalışma Süresi (test_1 süresi): {test_1_time:.2f} saniye")


    test_2_start=time.time()
    
    # TEST 2: Doğrudan LLM Yanıtı Gerektiren Sorgu (Araç kullanılmamalı)
    print("## 🚀 Test 2: Dogrudan LLM Yanıt Testi ##")
    user_query_2 = "Türkiye Cumhuriyeti'nin kurucusu kimdir?"

    
    # DÜZELTME: .run() yerine .achat() kullanıldı.
    response_llm = await agent.run(user_query_2)

    print(f"\n[AGENT YANITI]: {response_llm.response}")
    test_2_stop=time.time()
    test_2_time=test_2_stop-test_2_start
    print(f"⏱️ Çalışma Süresi (test_2 süresi): {test_2_time:.2f} saniye")
# Kodu başlat
if __name__ == "__main__":

    asyncio.run(main())