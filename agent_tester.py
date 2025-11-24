
import time
import os
import requests
import asyncio 
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent,ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.arxiv import ArxivToolSpec
import os
import subprocess
import whisper
import pandas as pd
import json
from PIL import Image
import base64
import io
import nest_asyncio
nest_asyncio.apply()

CLOUDFLARE_TUNNEL_URL = "..." 
OLLAMA_MODEL_ID = "gemma3:27b"
WEATHER_API = "..."


def WeatherInfoTool(location:str)->str:
        print("Using weather info tool...")
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

search_tool = DuckDuckGoSearchToolSpec()

def multiply_func(a:float,b:float)->float:
    print("Using mutply tool...")
    return a*b

multiply_tool=FunctionTool.from_defaults(
    multiply_func,
    name="multiply",
    description = "This tool is Multiply two numbers."
)

def add_func(a:float,b:float)->float:
    print("Using add tool...")
    return a+b

add_tool=FunctionTool.from_defaults(
    add_func,
    name="add_tool",
    description="This tool is add two numbers"
)

def sub_func(a:float,b:float)->float:
    print("Using subtract tool...")
    return a-b

subtract_tool=FunctionTool.from_defaults(
    sub_func,
    name="subtract_tool",
    description="This tool is subtract two numbers"
)

def div_func(a:float,b:float)->float:
    "This tool is divide two numbers but if second number is 0 this tool raise to error"
    print("Using divide tool...")
    if b==0:
       raise ValueError("Cannot divide by zero.") 
    else:
        return a/b
divide_tool=FunctionTool.from_defaults(
    div_func,
    name="divide_tool",
    description="This tool is divide two numbers but if second number is 0 this tool raise to error"
)


wiki_search_tool=WikipediaToolSpec()

archive_search_tool=ArxivToolSpec()

def transcribe_audio_whisper(audio_path:str)->str:
    print("Using audio transcriber tool...")
    model = whisper.load_model(
        "small")  # 'tiny', 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_path)
    return result["text"]

transcriber_tool=FunctionTool.from_defaults(
    transcribe_audio_whisper,
    name="mp3_transcript",
    description = "Extracts transcript from any voice file using Whisper"
)

def download_audio_from_youtube(url, output_path="audio.mp3"):
    
    subprocess.run([
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        output_path,
        url,
    ])


def download_video_from_youtube(url, output_path="video.mp4"):
    print("Using yt video download tool...")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo+bestaudio", "-o", output_path, url],
        capture_output=True,
        text=True,
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")





def youtube_transcript_func(url:str)->str:
    print("Using yt audio download tool...")
    audio_path = "audio.mp3"
    download_audio_from_youtube(url, audio_path)
    transcript = transcribe_audio_whisper(audio_path)
    os.remove(audio_path)
    return transcript

youtube_transcript_tool=FunctionTool.from_defaults(
    youtube_transcript_func,
    name = "youtube_transcript",
    description = "Extracts transcript from a YouTube video using Whisper"
)




def caption_image_func(image_path: str, prompt: str) -> str:
    print("Using image caption tool...")
    """
    Analyzes a local image file and generates a description or answer based on the given text prompt.
    
    Args:
        image_path (str): The local file path of the image to analyze (e.g., 'images/cat.jpg' or '/content/test.png').
        prompt (str): A question or instruction specifying what you want to learn about the image (e.g., 'What is in this picture?', 'How many cars do you see?').
        
    Returns:
        str: The textual response from the model regarding the image and the question.
    """
    global CLOUDFLARE_TUNNEL_URL
    global OLLAMA_MODEL_ID

    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = CLOUDFLARE_TUNNEL_URL + "/api/generate"

    payload = {
        "model": OLLAMA_MODEL_ID,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    }
    response = requests.post(url,
                             headers={"Content-Type": "application/json"},
                             json=payload)
    response.raise_for_status()  # Hatalı HTTP durum kodu varsa Exception atar

    data = response.json()

    if "response" in data:
        return data["response"]
    else:
        return "Image not recognized"

im_caption_tool=FunctionTool.from_defaults(
    caption_image_func,
    name = "image_captioner",
    description = (
        "Generates a detailed natural language description of the given image using a multimodal large language model "
        "via an Ollama server. Accepts a local image file (e.g., JPG or PNG) and a textual prompt describing "
        "what to look for in the image. The image is encoded in base64 and sent to the model for visual understanding."
    )
)


def file_download_func(task_id:str)->str:
    print("Using file download tool...")
    """
    Downloads the file corresponding to the given task_id and processes it according to its type.
    
    Args:
        task_id (str): The unique ID of the file to download (e.g., 'task_123').
        
    Returns:
        str: The file path, Excel/JSON content, or a success message.
    """


    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
    response = requests.get(url)

    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"')
    else:
        return "Unable to find a supported file type."

    file_path = os.path.join(".", filename)
    with open(file_path, "wb") as f:
        f.write(response.content)

    if filename.endswith(".mp3"):
        answer = f"The MP3 file was downloaded successfully. Saved at: {file_path}"
        return answer
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file_path)

        # İlk birkaç satırı düzgün formatta yazalım (çok büyükse tümünü yazmak istemeyebiliriz)
        df_preview = df.to_string(index=False)

        return (
            f"The file '{filename}' has been downloaded and read as an Excel spreadsheet.\n"
            f"Here is a preview of its contents:\n\n{df_preview}")

    elif filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return f"The file '{filename}' has been downloaded and its content is as follows:\n{json.dumps(data, indent=2)}"

    elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".PNG",
                            ".JPG", ".JPEG", ".BMP")):
        answer = (
            f"The image file was downloaded successfully. Saved at: {file_path}"
        )
        return answer
    else:
        os.remove(file_path)  # gereksiz dosyayı sil
        return "The downloaded file is not in a supported format."

file_download_tool=FunctionTool.from_defaults(
    file_download_func,
    name = "File_Download_Tool",
    description = (
        "Downloads a file from a Hugging Face-hosted URL using the given task_id.\n\n"
        "Supported file types and behaviors:\n"
        "- If .mp3 → Downloads the audio file and returns its saved path.\n"
        "- If .xlsx → Parses the Excel file and returns a readable text \n"
        "- If .json → Parses and returns the full JSON content in readable format.\n"
        "- If .jpg/.jpeg/.png/.bmp → Downloads the image file and returns its saved path.\n"
        "- For all other file types → Deletes the file and returns an unsupported format message."
    )
)


def download_video_from_youtube(url:str, output_path="video.mp4")->str:
    print("Using yt video download tool...")
    """
    Downloads a YouTube video from the provided URL using yt-dlp and saves it locally.
    
    Args:
        url (str): The full URL of the YouTube video to download (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').
        
    Returns:
        str: A message indicating success with the file path, or an error message if the download fails.
    """
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]",
            "--merge-output-format",
            "mp4",  # video+ses -> mp4
            "-o",
            output_path,
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    return output_path

yt_video_download_tool=FunctionTool.from_defaults(
    download_video_from_youtube,
    name = "youtube_video_download",
    description = (
        "This tool downloads a YouTube video from the provided URL using yt-dlp. "
        "After downloading, it returns the local path to the saved video file."
    )

)





def build_agent():
    global CLOUDFLARE_TUNNEL_URL
    global OLLAMA_MODEL_ID
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(
            "system_prompt.txt dosyası bulunamadı, varsayılan prompt kullanılacak."
        )
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "
    except Exception as e:
        print(f"system_prompt.txt okunurken hata oluştu: {e}")
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "

    model = Ollama(
        model=OLLAMA_MODEL_ID, 
        base_url=CLOUDFLARE_TUNNEL_URL, 
        context_window=8192,
        # verbose=True ekleyerek LiteLLM'in HTTP isteklerini gorebilirsiniz.
        request_timeout="3000"
    )
    # 1. Harici Paketleri (Specs) Hazırla
    # Bu paketler yüklü değilse boş liste döner, böylece kod çökmez.

    tool_list = [

        yt_video_download_tool,    
        file_download_tool,      
        im_caption_tool,           
        youtube_transcript_tool,  
        transcriber_tool,        
        weather_tool,             
        multiply_tool,             
        add_tool,                 
        subtract_tool,            
        divide_tool,               
              
    ]
    search_tools=search_tool.to_tool_list()
    search_tools.extend(archive_search_tool.to_tool_list())
    search_tools.extend(wiki_search_tool.to_tool_list())
    search_tools.extend(tool_list)

    Arxivangelist = ReActAgent( 
        tools=search_tools,
        llm=model,
        verbose=True, 
        system_prompt=system_prompt,
        # managed_agents=[video_agent],
        
    )
    return Arxivangelist

import time
import os
from PIL import Image
import pandas as pd

# --- ÖNCEKİ KODLARINIZIN BURADA OLDUĞUNU VARSAYIYORUZ ---
# (Yukarıdaki importlar, tool tanımları ve build_agent fonksiyonu)

# ---------------------------------------------------------
# TEST FONKSİYONU
# ---------------------------------------------------------

def create_dummy_files_for_test():
    """Testler için gerekli geçici dosyaları oluşturur."""
    print("🛠️ Test ortamı hazırlanıyor...")
    
    # 1. Vision Tool testi için basit bir resim oluştur
    if not os.path.exists("test_image.png"):
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save('test_image.png')
        print("   -> 'test_image.png' oluşturuldu.")

def run_agent_tests(agent):
    """
    Tanımlanan tüm araçları tek tek test eder ve raporlar.
    """
    create_dummy_files_for_test()
    
    # Test Senaryoları: (Test Adı, Agent'a Gönderilecek Prompt)
    test_scenarios = [
        (
            "Matematik (Çarpma)", 
            "Multiply 50 by 3. Return ONLY the result."
        ),
        (
            "Matematik (Bölme & Hata Kontrolü)", 
            "Divide 100 by 0. If it fails, divide 100 by 4."
        ),
        (
            "Hava Durumu (API)", 
            "What is the current weather in Ankara, Turkey?"
        ),
        (
            "Web Arama (DuckDuckGo)", 
            "Who is the current CEO of OpenAI? Search and tell me the name."
        ),
        (
            "Görsel Analiz (Vision Tool)", 
            "I have an image at path 'test_image.png'. Describe what you see in this image using the image_captioner tool."
        ),
        (
            "Wikipedia Arama", 
            "Search Wikipedia for 'Mustafa Kemal Atatürk' and summarize the first paragraph."
        ),
        #YouTube testi uzun sürebilir, test etmek isterseniz yorumu kaldırın
        (
           "YouTube Video İndirme",
           "Download this video: https://www.youtube.com/watch?v=BaW_jenozKc (Youtube 10 year video)"
        )
    ]

    print("\n" + "="*60)
    print("🤖 AJAN FONKSİYON TESTLERİ BAŞLIYOR")
    print("="*60)

    results = []

    for test_name, prompt in test_scenarios:
        print(f"\n🔹 TEST: {test_name}")
        print(f"   Soru: {prompt}")
        
        start_time = time.time()
        try:
            async def proc(prompt):
                # Agent'ı çalıştır
                response = await agent.run(prompt)
                return response.response
            response=asyncio.run(proc(prompt)) 
            duration = time.time() - start_time
            print(f"   ✅ BAŞARILI ({duration:.2f}sn)")
            print(f"   🤖 Cevap: {str(response)}")
            results.append((test_name, "BAŞARILI", duration))
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ BAŞARISIZ ({duration:.2f}sn)")
            print(f"   Hata: {e}")
            results.append((test_name, f"HATA: {str(e)}", duration))
        
        print("-" * 30)

    # ÖZET RAPOR
    print("\n" + "="*60)
    print("📊 TEST SONUÇ RAPORU")
    print("="*60)
    print(f"{'TEST ADI':<35} | {'DURUM':<10} | {'SÜRE':<10}")
    print("-" * 60)
    for name, status, dur in results:
        print(f"{name:<35} | {status:<10} | {dur:.2f}s")
    print("="*60)

    # Temizlik
    if os.path.exists("test_image.png"):
        os.remove("test_image.png")
        print("🧹 Geçici test dosyaları temizlendi.")

# ---------------------------------------------------------
# ÇALIŞTIRMA
# ---------------------------------------------------------

if __name__ == "__main__":
    # Agent'ı oluştur
    try:
        # Not: build_agent() fonksiyonunuzun parametrelerini düzelttiğinizi varsayıyorum
        # (additional_authorized_imports vb. kaldırılmalı)
        my_agent = build_agent()
        """
        async def proc(promt):
                # Agent'ı çalıştır
                response_llm = await my_agent.run(promt)
                return response_llm
        promt= "Türkiye Cumhuriyeti'nin kurucusu kimdir?" 
        response_llm= asyncio.run(proc(promt))           
        print(f"\n[AGENT YANITI]: {response_llm.response}")
        """
        
        # Testleri başlat
        run_agent_tests(my_agent)
        
    except Exception as e:
        print(f"Agent oluşturulurken hata oluştu: {e}")

