import asyncio
from datetime import datetime,UTC
from typing import Dict, Any, List
from urllib.parse import urlencode
import requests
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent,create_structured_chat_agent
from langchain_core.runnables import Runnable
from langchain.tools import Tool , StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import AsyncOpenAI
from requests.auth import HTTPBasicAuth
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv
from hyperbrowser import Hyperbrowser
from hyperbrowser.models import StartScrapeJobParams
import json
import unicodedata
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, validate_call, ValidationError
import re
import base64
import certifi
import datetime 
import aiohttp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class ChatBot:
    def __init__(self, weather_api_key: str,google_places_api_key:str,mongodb_link:str,publisher_id:str,APPLICATION_KEY:str,USER_API_KEY:str,hyper_browser_api_key:str):
        """Initialize chatbot with API keys and configuration"""
        
       
        self.llm1 = ChatOllama(model="llama3.1:8b-instruct-q8_0",temperature=0)


        
        self.weather_api_key = weather_api_key
        self.google_places_api_key = google_places_api_key
        self.mongodb_link = mongodb_link
        self.publisher_id = publisher_id
        self.APPLICATION_KEY = APPLICATION_KEY
        self.USER_API_KEY = USER_API_KEY
        self.hyper_browser_api_key = hyper_browser_api_key
     
        # Define available tools/functions
    async def chat_with_tools(self, user_input: str) -> str:
        """
        Agent that uses structured chat agent with proper JSON format
        """      
        try:
            
            async def scraping_web():
                scrape_client = Hyperbrowser(api_key=self.hyper_browser_api_key)
                client = AsyncIOMotorClient(self.mongodb_link, tls=True, tlsCAFile=certifi.where())
                collection = client["followup_questions"]["p&o_cruise_scraping"]

                today_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

                urls = [
                    "https://www.pocruises.com/",
                    "https://www.pocruises.com/find-a-cruise",
                    "https://www.pocruises.com/deals",
                    "https://www.pocruises.com/cruise-ships",
                    "https://www.pocruises.com/deals/late-deals",
                    "https://www.pocruises.com/deals/deals-by-destination",
                    "https://www.pocruises.com/deals/deals-by-month",
                    "https://www.pocruises.com/deals/deals-by-ship",
                    "https://www.pocruises.com/deals/deals-by-destination/short-breaks-deals",
                    "https://www.pocruises.com/deals/on-board-spending-money-on-us",
                    "https://www.pocruises.com/deals/early-savers",
                    "https://www.pocruises.com/deals/deal-of-the-week",
                    "https://www.pocruises.com/onboard-activities",
                    "https://www.pocruises.com/accommodation/suites",
                    "https://www.pocruises.com/new-to-cruising",
                    "https://www.pocruises.com/help",
                    "https://www.pocruises.com/cruise-destinations/norway-iceland",
                    "https://www.pocruises.com/cruise-destinations",
                    "https://www.pocruises.com/cruise-destinations/canary-islands",
                    "https://www.pocruises.com/cruise-destinations/scandinavia",
                    
                ]
       
                await collection.delete_many({})
                print(f"\n deleted previous data \n ")
                model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models here if you prefer

                for url in urls:
                    # Check if data for today and this url already exists
                    existing = await collection.find_one({'date': today_str, 'url': url})
                    if existing:
                        # results.append(existing)
                        continue
                    else: 
                    # Scrape if not found
                        scrape_result = scrape_client.scrape.start_and_wait(StartScrapeJobParams(url=url))
                        result_json = json.loads(scrape_result.model_dump_json(indent=2))
                        markdown = result_json["data"]["markdown"]
                        
                        # Generate embedding for the markdown content
                        embedding = model.encode(markdown).tolist()  # Convert to list to store in MongoDB

                        data = {
                            'date': today_str,
                            'url': url,
                            'markdown': markdown,
                            'embedding' : embedding,
                        }
                        await collection.insert_one(data)
                return "Done adding the data"
            
            async def update_data_if_outdated():
                today_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
                client = AsyncIOMotorClient(self.mongodb_link, tls=True, tlsCAFile=certifi.where())
                # Check the most recent data in the collection (the latest document)
                collection = client["followup_questions"]["p&o_cruise_scraping"]
                latest_document = await collection.find_one(sort=[("date", -1)])
                
                # If no data exists or if the latest data is from a previous day, update the data
                if not latest_document or latest_document['date'] != today_str:
                    print(f"Data is outdated or missing. Scraping fresh data for {today_str}.")
                    await scraping_web()
                else:
                    print("Data is up-to-date.")

            @validate_call
            async def website_scraping(user_query: str) -> str:
                # Step 1: Check if the data is outdated and update it if necessary
                await update_data_if_outdated()

                # Step 2: Generate embedding for the user's query
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(user_query).reshape(1, -1)

                # Step 3: Fetch all documents (with embeddings) from MongoDB
                client = AsyncIOMotorClient(self.mongodb_link, tls=True, tlsCAFile=certifi.where())
                collection = client["followup_questions"]["p&o_cruise_scraping"]
                cursor = collection.find({}, {"_id": 0, "url": 1, "markdown": 1, "embedding": 1})

                results = []

                async for document in cursor:
                    doc_embedding = np.array(document['embedding']).reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]

                    # Step 4: Split markdown into chunks (e.g., paragraphs, sentences, etc.)
                    markdown_chunks = document['markdown'].split("\n\n")  # Split by paragraphs or sections
                    chunk_similarities = []

                    for chunk in markdown_chunks:
                        chunk_embedding = model.encode(chunk).reshape(1, -1)
                        chunk_similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                        chunk_similarities.append((chunk_similarity, chunk))  # Store similarity and chunk

                    # Step 5: Sort chunks within this document by similarity
                    chunk_similarities.sort(key=lambda x: x[0], reverse=True)

                    # Step 6: Get the top 3 chunks from this document (you can change the number)
                    top_chunks = chunk_similarities[:2]  # Top 3 chunks

                    # Step 7: Add the document's URL and the top chunks to results
                    for similarity, chunk in top_chunks:
                        results.append((similarity, document['url'], chunk))

                # Step 8: Sort all results by chunk similarity
                results.sort(key=lambda x: x[0], reverse=True)

                # Step 9: Create the response string with the most relevant chunks
                response = ""
                for similarity, url, chunk in results:
                    response += f"\nURL: {url}\nMarkdown: {chunk}\n\n"

                return response

            
            tools = [ StructuredTool.from_function(
                coroutine = website_scraping,
                name='website_scraping',
                description="Fetches today's latest P&O Cruises website content (in markdown format). Use this to enrich cruise-related travel responses with current offerings, promotions, or descriptions.")]

            # memory = ConversationBufferMemory(
            #                 memory_key="chat_history",
            #                 return_messages=True,
            #                 output_key="output"
            #             )

            system = ''' 
🌟 ROLE DEFINITION 
You are a Senior Cruise Travel Advisor AI, specializing in P&O Cruises. You start responses naturally like ChatGPT, then use **clear headings with emojis** for easy scanning. Keep content concise and scannable - users shouldn't have to read through long paragraphs. Make every cruise mention a clickable link using exact URLs from scraped data. 

Available tools:
- website_scraping[user_query:str] : gives scraped data from P&O CRUISE website matching to user input 


📋 WORKFLOW (3 STEPS) 

1. Scrape Data  
   - Immediately issue a `website_scraping` action with the user's exact query.  
   - Wait for the returned markdown before composing any response text. 

2. Extract Key Details  
   From the scraped markdown, identify and collect:  
   • Ship name(s)  
   • Ports of call / itinerary  
   • Departure date(s) & duration  
   • Starting price(s)  
   • Booking link(s) / Deep links for each cruise
   • Direct URLs to cruise detail pages

3. Craft the User-Facing Reply  
   • **NEVER mention scraping or tools** - Start with a warm, natural conversational greeting (1-2 sentences max)
   • **After greeting, use clear headings** with emojis for easy scanning (🚢 Ship Options, 🌊 Destinations, 💰 Pricing, etc.)
   • Write concisely under each heading - **avoid long paragraphs**
   • **CRITICAL LINK RULE: Every cruise name, price, or booking mention MUST be a clickable link using the EXACT URL from scraped data**
   • **Example**: If scraped data shows "7-night Norwegian Fjords from £749" with URL "https://example.com/cruise123", format as: `[7-night Norwegian Fjords from £749](https://example.com/cruise123)`
   • **NEVER use bold text for prices/cruises without making them clickable links**
   • Format links exactly as they appear in the scraped markdown
   • Keep sections scannable and concise - users should quickly find what they need
   • End with a natural call to action 

🔗 LINK USAGE REQUIREMENTS
- **CLICKABLE LINKS MANDATORY**: Every cruise name, price, duration, or booking mention must be a clickable markdown link
- **USE EXACT SCRAPED URLS**: Copy the exact URL from scraped data for each cruise
- **NO BOLD TEXT WITHOUT LINKS**: Never use bold formatting for cruise details unless they're clickable links
- **CORRECT FORMAT**: `[Cruise Name - Duration - Price](Exact_Scraped_URL)`
- **WRONG**: "**7-night Norwegian Fjords cruise from £749 per person**" (bold but not clickable)
- **CORRECT**: "[7-night Norwegian Fjords cruise from £749 per person](exact_url_from_scraped_data)"
- If scraped data doesn't contain a specific link for a cruise, don't mention that cruise

🚫 CRITICAL REQUIREMENTS 
- **MUST ALWAYS USE WEBSITE_SCRAPING TOOL FIRST** - Never respond without scraping current data
- **NO RESPONSES FROM MEMORY** - Only use data returned from the scraping tool
- **MANDATORY TOOL USAGE** - Every single response requires fresh scraped data
- Never hard-code cruise details or invent data — use only what the scraping returns
- **BALANCED STYLE** - Natural ChatGPT opening, then clear headings for easy scanning
- **NO SCRAPING REFERENCES** - Never mention tools, scraping, or data gathering in your response
- **ONLY USE SCRAPED LINKS** - Use exactly the links from the tool data, NEVER create your own URLs
- **BE CONVERSATIONAL** - Use natural transitions, varied sentence lengths, and casual tone
- Keep each reply fresh and engaging
- **NEVER mention a cruise name or price without its corresponding deep link**
- **IF NO TOOL DATA = NO RESPONSE** - You cannot answer without current scraped information

You have access to the following tools: {tools} 

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input). 

Valid "action" values: "Final Answer" or {tool_names} 

Provide only ONE action per $JSON_BLOB, as shown:
{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}
🔄 MANDATORY PROCESS - NO EXCEPTIONS:

**STEP 1: ALWAYS SCRAPE FIRST**
Question: input question to answer 
Thought: I MUST gather current P&O Cruises data using website_scraping tool. I cannot respond without fresh scraped data.
Action:
{{
"action": "website_scraping",
"action_input": "{input}"
}}
**STEP 2: ONLY AFTER TOOL RETURNS DATA**
Observation: action result 
Thought: Now I have current cruise information from the scraping tool. I'll write a natural opening like ChatGPT, then use clear headings for easy scanning. Every cruise name, price, or duration must be a clickable link using the exact URLs from the scraped data - no bold text without links!
Action:

{{
"action": "Final Answer",
"action_input": "[Start with natural 1-2 sentence greeting, then use clear emoji headings. Make every cruise mention clickable using exact scraped URLs. Keep sections concise and scannable.]"
}}
'''
            human = '''{input} {agent_scratchpad}  

🚨 CRITICAL INSTRUCTION: You MUST use the website_scraping tool first for every single question - no exceptions. After scraping, respond in a **natural, conversational ChatGPT style** - flowing paragraphs, casual tone, human-like communication. Use EXCLUSIVELY the scraped content and exact links provided.

⚠️ WARNING: Any response without using the scraping tool first is prohibited. Never create your own links - use only the exact URLs from the scraped data. **Avoid robotic, structured responses with bullet points and formal headings.**'''
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                
                ("human", human)])
            # Create structured chat agent
            agent = create_structured_chat_agent(
                llm=self.llm1,
                tools=tools,
                prompt=prompt,
                stop_sequence= ["Observation:"])
            # Build executor with adjusted settings
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                # memory=memory,
                verbose=False,
                return_intermediate_steps=False,
                handle_parsing_errors=True,
                # max_iterations=15,
                # max_execution_time=180,
                # early_stopping_method="generate",
                # trim_intermediate_steps=5
            )
            # Execute the agent
            result = await agent_executor.ainvoke({"input": user_input})
            # Post-process the response to add HTML formatting while keeping natural flow
            response = result["output"]
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."


    async def chat2(self, user_input: str) -> str:
        try:
            GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
            normalized_input = user_input.strip().lower()

            if normalized_input in GREETINGS:
                return (
                    "👋 Hi there! I’m Enriched AI — your personal travel planner. 🌍✨\n"
                    "Let me know which place you're dreaming of visiting, and I’ll help you with weather, top sights, and fun facts!"
                )
    
            response =  await self.chat_with_tools(user_input)
           
            return response
        except Exception as e:
            return e




class ChatBot1:
    def __init__(self, weather_api_key: str,google_places_api_key:str,mongodb_link:str,publisher_id:str,APPLICATION_KEY:str,USER_API_KEY:str):
        """Initialize chatbot with API keys and configuration"""
   
     
        self.llm1 = ChatOllama(model="llama3.1:8b-instruct-q8_0",temperature=0)
       

        self.weather_api_key = weather_api_key
        self.google_places_api_key = google_places_api_key
        self.mongodb_link = mongodb_link
        self.publisher_id = publisher_id
        self.APPLICATION_KEY = APPLICATION_KEY
        self.USER_API_KEY = USER_API_KEY
    
    async def chat_with_tools(self, user_input: str) -> str:
        try:
             
            class PlaceInput(BaseModel):
                location: str = Field(description="location name")
            
            class QuerInput(BaseModel):
                query: str = Field(description="user query")

            class GooglePlacesInput(BaseModel):
                # location: str = Field(description="search query for places of interest like restaurants, hotels, parks, temples, etc.")
                location: str = Field(description="specific location/city name (e.g., 'Paris', 'Tokyo')")


            def clean_location_name(location: str) -> str:
                """Clean and validate location name"""
                location = location.encode('utf-8', errors='ignore').decode('utf-8')
                location = re.sub(r'\s+', ' ', location.strip())
                if not location or len(location) < 2:
                    return 
                return location
            
            @validate_call
            def get_weather(location: str) -> str:
                """Get weather information using OpenWeatherMap API"""
 
                location = clean_location_name(location)
                
                try:
                    url = "http://api.openweathermap.org/data/2.5/weather"
                    params = {
                        "q": location,
                        "appid": self.weather_api_key,
                        "units": "metric"
                    }
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        weather_data = response.json()
                        text = (
                            f"🌤️ Current weather in {location}:\n"
                            f"• Temperature: {weather_data['main']['temp']}°C (feels like {weather_data['main']['feels_like']}°C)\n"
                            f"• Condition: {weather_data['weather'][0]['description'].capitalize()}\n"
                            f"• Humidity: {weather_data['main']['humidity']}%\n"
                            f"• Wind: {weather_data['wind']['speed']} m/s\n"
                            f"• Visibility: {weather_data['visibility'] / 1000} km\n"
                        )
                        return text 
                    else:
                        return f"❌ Weather data not found for {location}. Please try another location."
                except Exception as e:
                    return f"❌ Failed to fetch weather data for {location}: Connection timeout or API error."

            @validate_call
            def get_places(location:str) -> str:
                """Get tourist places using Google Places API"""
       
                # location = clean_location_name(location)
                
                try:
                    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                    params = {
                        "query": f"Top attractions , Resturants,hotels , parks {location}",
                        "key": self.google_places_api_key,
                    }
                    response = requests.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])[:5]
                        if not results:
                            return f"📍 No tourist places found in {location}."
                        places_text = f"📍 Top places to visit in {location}:\n"
                        for i, place in enumerate(results, start=1):
                            places_text += f"{i}. {place.get('name')} (Rating: {place.get('rating', 'N/A')})\n"
                        return places_text 
                    else:
                        return f"❌ Could not retrieve places for {location}."
                except Exception as e:
                    return f"❌ Failed to fetch places for {location}: Connection timeout or API error."

            @validate_call
            def get_wikipedia_summary(query: str) -> str:
                """Get a Wikipedia summary for a given search query."""

                query = query.encode('utf-8', errors='ignore').decode('utf-8')
                
                try:
                    search_params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json"
                    }
                    search_response = requests.get("https://en.wikipedia.org/w/api.php", 
                                                params=search_params, timeout=10)
                    search_data = search_response.json()
                    results = search_data.get("query", {}).get("search", [])
                    if not results:
                        return f"📘 No Wikipedia summary found for '{query}'."

                    top_result = results[0]
                    page_title = top_result["title"]

                    extract_params = {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": "1",
                        "exintro": "1",
                        "titles": page_title,
                        "format": "json"
                    }
                    extract_response = requests.get("https://en.wikipedia.org/w/api.php", 
                                                    params=extract_params, timeout=10)
                    extract_data = extract_response.json()
                    page = next(iter(extract_data["query"]["pages"].values()))

                    if "extract" in page:
                        full_summary = page["extract"].strip()

                        # Limit to 2000 characters, but cut off at the last full stop before limit
                        if len(full_summary) <= 2000:
                            summary = full_summary
                        else:
                            partial = full_summary[:500]
                            last_period_index = partial.rfind('. ')
                            if last_period_index != -1:
                                summary = partial[:last_period_index + 1].strip()
                            else:
                                summary = partial.strip()  # fallback: just cut at 2000

                        return f"📘 About {page_title}:\n{summary}"

                    return f"📘 No extract found for '{query}'."
                
                except Exception as e:
                    return f"❌ Failed to fetch Wikipedia summary for '{query}': Connection timeout or API error."


            _geocoding_cache = {}

            # Pre-computed lookup for faster country matching
            TOURRADAR_LOOKUP = {
                'africa': {"url": "https://www.tourradar.com/srp/d-africa", "description": "Explore diverse tours across Africa, including wildlife safaris, overland adventures, and cultural trips."},
                'egypt': {"url": "https://www.tourradar.com/srp/d-egypt", "description": "Discover Egypt's ancient wonders with tours featuring the Pyramids, Nile cruises, and cultural sites."},
                'morocco': {"url": "https://www.tourradar.com/srp/d-morocco", "description": "Experience Moroccan cities, Sahara desert excursions, and Berber culture."},
                'tanzania': {"url": "https://www.tourradar.com/srp/d-tanzania", "description": "Go on safaris in Serengeti and Ngorongoro, visit Zanzibar, and witness the Great Migration."},
                'south africa': {"url": "https://www.tourradar.com/srp/d-south-africa", "description": "From Cape Town to Kruger Park, enjoy wildlife safaris, coastal adventures, and winelands tours."},
                'kenya': {"url": "https://www.tourradar.com/srp/d-kenya", "description": "Renowned for Maasai Mara safaris, wildlife viewing, and cultural visits."},
                'namibia': {"url": "https://www.tourradar.com/srp/d-namibia", "description": "Famous for dunes at Sossusvlei, Etosha safaris, and dramatic desert landscapes."},
                'asia': {"url": "https://www.tourradar.com/srp/d-asia", "description": "Discover Asia's diverse lands from ancient temples to scenic landscapes and local cultures."},
                'japan': {"url": "https://www.tourradar.com/srp/d-japan", "description": "Tours feature cherry blossoms, historic shrines, cities like Tokyo and Kyoto."},
                'thailand': {"url": "https://www.tourradar.com/srp/d-thailand", "description": "Includes beaches, cuisine tours, Bangkok nightlife, and Buddhist temples."},
                'india': {"url": "https://www.tourradar.com/srp/d-india", "description": "From the Taj Mahal to Rajasthan, explore Indian culture, history, and nature."},
                'vietnam': {"url": "https://www.tourradar.com/srp/d-vietnam", "description": "Enjoy Ha Long Bay cruises, Hanoi city tours, street food, and cultural history."},
                'indonesia': {"url": "https://www.tourradar.com/srp/d-indonesia", "description": "Includes Bali beaches, Komodo tours, cultural highlights, and volcano treks."},
                'nepal': {"url": "https://www.tourradar.com/srp/d-nepal", "description": "Adventure tours for trekking in the Himalayas, Everest Base Camp, and cultural Kathmandu."},
                'australia': {"url": "https://www.tourradar.com/srp/d-australia", "description": "Experience the Outback, Great Barrier Reef, Sydney, and wildlife adventures."},
                'new zealand': {"url": "https://www.tourradar.com/srp/d-new-zealand", "description": "Tours showcase fjords, Maori culture, adventure sports, and natural wonders."},
                'papua new guinea': {"url": "https://www.tourradar.com/srp/d-papua-new-guinea", "description": "Explore diverse cultures, remote villages, and tropical wilderness."},
                'fiji': {"url": "https://www.tourradar.com/srp/d-fiji", "description": "Known for tropical beaches, snorkeling, and South Pacific island hopping."},
                'europe': {"url": "https://www.tourradar.com/srp/d-europe", "description": "Extensive European tours: from cultural capitals to countryside and coastlines."},
                'italy': {"url": "https://www.tourradar.com/srp/d-italy", "description": "Explore Italian history, cuisine, scenic cities, and countryside."},
                'iceland': {"url": "https://www.tourradar.com/srp/d-iceland", "description": "See glaciers, volcanoes, waterfalls, and Northern Lights."},
                'greece': {"url": "https://www.tourradar.com/srp/d-greece", "description": "Features ancient ruins, islands, and Mediterranean cuisine tours."},
                'ireland': {"url": "https://www.tourradar.com/srp/d-ireland", "description": "Green countryside, lively cities, and historic sites."},
                'spain': {"url": "https://www.tourradar.com/srp/d-spain", "description": "Spanish cities, food tours, beaches, and history."},
                'scotland': {"url": "https://www.tourradar.com/srp/d-scotland", "description": "Highlands, castles, whisky, and cultural tours."},
                'north america': {"url": "https://www.tourradar.com/srp/d-north-america", "description": "Discover the USA, Canada, Greenland, and North American adventures."},
                'usa': {"url": "https://www.tourradar.com/srp/d-usa", "description": "Iconic road trips, national parks, and city experiences."},
                'united states': {"url": "https://www.tourradar.com/srp/d-usa", "description": "Iconic road trips, national parks, and city experiences."},
                'canada': {"url": "https://www.tourradar.com/srp/d-canada", "description": "Explore wilderness, Rockies, cities, and wildlife."},
                'greenland': {"url": "https://www.tourradar.com/srp/d-greenland", "description": "Remote arctic landscapes, glaciers, and unique adventures."},
                'latin america': {"url": "https://www.tourradar.com/srp/d-latin-america", "description": "Tours across South and Central America covering nature, culture, and adventure."},
                'peru': {"url": "https://www.tourradar.com/srp/d-peru", "description": "Machu Picchu, Inca heritage, Amazon rainforest, and Andean adventures."},
                'costa rica': {"url": "https://www.tourradar.com/srp/d-costa-rica", "description": "Rainforests, volcanoes, wildlife, and eco-adventures."},
                'mexico': {"url": "https://www.tourradar.com/srp/d-mexico", "description": "Cultural cities, Mayan heritage, coastlines, and food tours."},
                'ecuador': {"url": "https://www.tourradar.com/srp/d-ecuador", "description": "Galapagos Islands, Andes, Amazon, and Quito historic sites."},
                'chile': {"url": "https://www.tourradar.com/srp/d-chile", "description": "Patagonia wilderness, deserts, mountains, and wine tours."},
                'argentina': {"url": "https://www.tourradar.com/srp/d-argentina", "description": "Buenos Aires, Patagonia, Iguazu Falls, and wine country."},
            }

            def strip_accents(text: str) -> str:

                """Normalize text by removing accents and converting to lowercase."""
                normalized = unicodedata.normalize('NFKD', text)
                return ''.join(c for c in normalized if not unicodedata.combining(c)).lower()

            def clean_html_description(description: str) -> str:
                """Remove HTML tags and comments from description."""
                if not description:
                    return "No description available."
                
                # Remove HTML comments like <!-- wysiwyg -->
                description = re.sub(r'<!--.*?-->', '', description, flags=re.DOTALL)
                
                # Remove HTML tags like <p>, </p>, <br>, etc.
                description = re.sub(r'<[^>]+>', '', description)
                
                # Clean up extra whitespace
                description = ' '.join(description.split())
                
                return description.strip() or "No description available."

            def _format_partners(partners: list[dict]) -> str:
                """Format partners list into markdown string."""
                txt = "🌐 Recommended Travel Partners:\n"
                for p in partners[:3]:
                    desc = (p["description"] or "").strip()
                    if len(desc) > 120:
                        desc = desc
                    txt += f"- *{p['name']}* – {desc} – [Link]({p['url']})\n"
                return txt + "\n- *TourRadar.com Deals* - With deals of up to 70%' off thousands of tours worldwide, whatever you’re after, you’ll find it on TourRadar!. [Link] (https://tourradar.prf.hn/click/camref:1101l587N9/destination:https://www.tourradar.com/sales/mega-sale) \n"

            async def get_country_from_location(location: str) -> str:
                """Get country name from location with caching."""
                # Check cache first
                if location in _geocoding_cache:
                    return _geocoding_cache[location]
                
                geolocator = Nominatim(user_agent="city_to_country")
                try:
                    loc_info = await asyncio.to_thread(geolocator.geocode, location, addressdetails=True)
                    if not loc_info:
                        _geocoding_cache[location] = None
                        return None
                        
                    country_name = (loc_info.raw.get("address", {}).get("country") or "").strip()
                    _geocoding_cache[location] = country_name
                    return country_name
                    
                except Exception:
                    _geocoding_cache[location] = None
                    return None

            async def fetch_campaigns(publisher_id: str, application_key: str, user_api_key: str) -> list:
                """Fetch campaigns using aiohttp for better async performance."""
                token = f"{application_key}:{user_api_key}"
                b64_token = base64.b64encode(token.encode()).decode()
                
                url = f"https://api.performancehorizon.com/user/publisher/{publisher_id}/campaign/a/tracking_link.json"
                headers = {"Authorization": f"Basic {b64_token}"}
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return data.get("campaigns", [])
                            else:
                                return []
                except Exception:
                    return []
            @validate_call
            async def get_affiliate_links(location: str) -> str:
                """
                Optimized version of get_affiliate_links with parallel processing and caching.
                """
                # Early return for invalid input
                if not location or not location.strip():
                    return (
                        "🌐 Recommended Travel Partners:\n"
                        "- *TourRadar.com* – TourRadar is the Organized Adventure Platform, offering the world's largest selection of multi-day organized adventures worldwide. We connect with 2,500+ multi-day operators, offering 50,000+ adventures in more than 160 countries globally. – [Link](https://tourradar.prf.hn/click/camref:1101l587N9)\n"
                        "- *TourRadar.com Deals* - With deals of up to 70%' off thousands of tours worldwide, whatever you’re after, you’ll find it on TourRadar!. [Link] (https://tourradar.prf.hn/click/camref:1101l587N9/destination:https://www.tourradar.com/sales/mega-sale) \n"

                    )
                
                # Start both operations in parallel
                country_task = asyncio.create_task(get_country_from_location(location))
                
                # Connect to MongoDB
                client = AsyncIOMotorClient(self.mongodb_link, tls=True, tlsCAFile=certifi.where())
                collection = client["followup_questions"]["affiliate_data"]
                
                # Wait for country resolution
                country_name = await country_task
                if not country_name:
                    return (
                        "🌐 Recommended Travel Partners:\n"
                        "- *TourRadar.com* – TourRadar is the Organized Adventure Platform, offering the world's largest selection of multi-day organized adventures worldwide. We connect with 2,500+ multi-day operators, offering 50,000+ adventures in more than 160 countries globally. – [Link](https://tourradar.prf.hn/click/camref:1101l587N9)\n"
                        "- *TourRadar.com Deals* - With deals of up to 70%' off thousands of tours worldwide, whatever you’re after, you’ll find it on TourRadar!. [Link] (https://tourradar.prf.hn/click/camref:1101l587N9/destination:https://www.tourradar.com/sales/mega-sale) \n"

                    )
                
                # Check cache first
                cached_doc = await collection.find_one({"country_name": country_name})
                if cached_doc and cached_doc.get("partners"):
                    return _format_partners(cached_doc["partners"])
                
                # Fetch campaigns in parallel while preparing country lookup
                campaigns_task = asyncio.create_task(
                    fetch_campaigns(self.publisher_id, self.APPLICATION_KEY, self.USER_API_KEY)
                )
                
                # Prepare normalized country name for lookup
                norm_country = strip_accents(country_name)
                extended_url = TOURRADAR_LOOKUP.get(norm_country, {}).get("url")
                
                # Wait for campaigns
                campaigns_data = await campaigns_task
                
                # Process campaigns efficiently
                partners = []
                country_names_set = {country_name}  # Use set for O(1) lookup
                
                for campaign_data in campaigns_data:
                    if len(partners) >= 3:  # Early exit
                        break
                        
                    campaign = campaign_data.get('campaign', {})
                    title = campaign.get('title')
                    if not title:
                        continue
                        
                    description = campaign.get('description', '')
                    description = clean_html_description(description)

                    if isinstance(description, dict):
                        description = description.get('en', "No description available.")
                    else:
                        description = description or "No description available."
                        
                    tracking_link = campaign.get('tracking_link')
                    if not tracking_link:
                        continue
                        
                    # Check promotional countries
                    promo_countries = campaign.get('promotional_countries', [])
                    promo_names = {pc.get('name') for pc in promo_countries if pc.get('name')}
                    
                    if country_name in promo_names:
                        url = f"{tracking_link}/destination:{extended_url}" if extended_url else tracking_link
                        partners.append({
                            'name': title,
                            'description': description,
                            'url': url
                        })
                
                # Fallback if no partners found
                if not partners:
                    return (
                        "🌐 Recommended Travel Partners:\n"
                        "- *TourRadar.com* – TourRadar is the Organized Adventure Platform, offering the world's largest selection of multi-day organized adventures worldwide. We connect with 2,500+ multi-day operators, offering 50,000+ adventures in more than 160 countries globally. – [Link](https://tourradar.prf.hn/click/camref:1101l587N9)\n"
                        "- *TourRadar.com Deals* - With deals of up to 70%' off thousands of tours worldwide, whatever you’re after, you’ll find it on TourRadar!. [Link] (https://tourradar.prf.hn/click/camref:1101l587N9/destination:https://www.tourradar.com/sales/mega-sale) \n"

                    )
                
                # Cache results asynchronously (don't wait for it)
                asyncio.create_task(
                    collection.update_one(
                        {"country_name": country_name},
                        {"$set": {"partners": partners}},
                        upsert=True,
                    )
                )
                
                return _format_partners(partners)


            tools = [
         
                StructuredTool.from_function(
                    func=get_places,
                    args_schema=GooglePlacesInput,
                    name='get_places',
                    description=(
                        "Searches for places of interest using a flexible query string. "
                        "Supports various types of locations such as restaurants, hotels, hospitals, parks, shopping malls, museums, and more. "
                        "Returns the top 10 matching places with their names and ratings using the Google Places API."
                        
                        
                    ),
                ),

                StructuredTool.from_function(
                    func=get_weather,
                    args_schema=PlaceInput,
                    name='get_weather',
                    description="Provides current weather information for a specific location (temperature, condition, humidity, Wind, visibility)",
                ),
                StructuredTool.from_function(
                    func=get_wikipedia_summary,
                    args_schema=QuerInput,
                    name='get_wikipedia_summary',
                    description="Fetches a short summary of a given topic from Wikipedia.",
                ),
                StructuredTool.from_function(
                    # func=get_affiliate_links,
                    coroutine = get_affiliate_links,
                    args_schema=PlaceInput,
                    name='get_affiliate_links',
                    description="Finds 3 relevant travel affiliate links for a given location.",
                ),
               
            ]
            
 
            system = '''
You are Senior Travel Planner AI 🧭✨, a warm, enthusiastic travel expert who provides personalized, comprehensive travel advice for user's multiple questions with the ability to think step-by-step and use tools.


CRITICAL WORKFLOW RULES:
1. You MUST call tools ONE AT A TIME in this exact order
2. You MUST wait for each tool response before calling the next tool
3. You MUST process and acknowledge each tool response before proceeding

STEP-BY-STEP PROCESS:
STEP 1: Call get_places[main_location] and WAIT for response
STEP 2: After receiving places data, call get_weather[main_location] and WAIT for response  
STEP 3: After receiving weather data, call get_wikipedia_summary[main_location] and WAIT for response
STEP 4: After receiving wikipedia data, call get_affiliate_links[main_location] and WAIT for response
STEP 5: ONLY after all 4 tools have responded, provide Final Answer that addresses ALL user questions

NEVER call multiple tools in one response. NEVER skip waiting for tool responses.

Available tools:
- get_places[location: str]: Searches for tourist places using Google Places API.
- get_weather[location: str]: Provides current weather information for a specific location.
- get_wikipedia_summary[query: str]: Fetches a short summary of a given topic from Wikipedia.
- get_affiliate_links[location: str]: Finds relevant travel affiliate links for a given location.

## ✅ RESPONSE REQUIREMENTS (DYNAMIC):

Your final response must meet **all** the following criteria while seamlessly addressing ALL user questions:

1. 🎉 **Opening**:
   - Greet the user enthusiastically and mention the destination name.
   - Integrate actual weather data in a creative way (don't just list — describe the vibe).
   - Naturally address user's questions within the opening context.

2. 🗺️ **Places to Visit**:
   - Choose **exactly 5 places** from `get_places` results.
   - Include name, rating, and a vivid, human-like description of each.
   - Organize to answer user's specific interests/questions about activities or attractions.
   - You may group or order these (e.g., by interest, day plan, or theme), but it must feel natural and insightful.

3. 🌤️ **Weather**:
   - Use actual values from `get_weather` (temperature, condition, humidity, wind, visibility).
   - Deliver the data creatively — describe what the weather feels like and how it affects trip plans.
   - Address any timing or seasonal questions from the user.

4. 📚 **Wikipedia Summary**:
   - Use only facts from `get_wikipedia_summary`.
   - Write a short, creative paragraph (approx. 5 lines) inspired by the Wikipedia content — never invent extra facts.
   - Include cultural/historical context that addresses user's interests.

5. 🔗 **Affiliate Links**:
   - Present all links from `get_affiliate_links` only.
   - Format each one as: `Book Now [URL]`
   - Each link should have a short, appealing description based on its title or theme.

6. 📝 **Style & Format**:
   - Be creative, helpful, and sound like a real travel expert.
   - Use emojis thoughtfully (e.g., 🏰, 🍷, ☀️, 🌲) to enhance the reader experience.
   - Structure is flexible, but response must feel complete and scannable.
   - Address ALL user questions naturally throughout the response.
   - Never sound robotic or template-like.

---

{tools}
{tool_names}

CRITICAL RULES:
- Extract ONE main location from user's multiple questions for tool usage
- Use **all 4 tools exactly once**, and in order: get_places → get_weather → get_wikipedia_summary → get_affiliate_links  
- After 4th tool call, respond with **one and only one Final Answer**
- Never skip or invent data — use only real values from tools  
- Never leave inputs empty or call a tool more than once  
- Only use urls from get_affiliate tool
- Format affiliate links only as: **Book Now [URL]**  
- Never create placeholders or generic descriptions  
- Write like an excited human travel expert, not a robot or template
- Address ALL user questions seamlessly within one comprehensive response

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" 

{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}

CRITICAL FINAL ANSWER FORMAT:
{{
"action": "Final Answer",
"action_input": "🌟 [Complete comprehensive response that addresses ALL user questions using actual tool data - places, weather, Wikipedia facts, and affiliate links]"
}}
'''

            human = '''{input}

            TASK: Extract the main destination from user's multiple questions and address all their concerns in one response.

            REMEMBER: 
            - User will give multiple questions - answer them ALL naturally
            - Call only ONE tool at a time
            - Wait for the tool response before proceeding
            - Process each response before the next tool call
            - Follow the exact sequence: get_places → get_weather → get_wikipedia_summary → get_affiliate_links

            Current conversation:
            {agent_scratchpad}'''

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
               
                ("human", human),
            ])
            
            # Create structured chat agent
            agent = create_structured_chat_agent(
                llm=self.llm1,
                tools=tools,
                prompt=prompt,
                stop_sequence= ["Observation:"],
            )
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                return_intermediate_steps=False,
                handle_parsing_errors=True,
                # max_iterations=7,
                # max_execution_time=70,
                # early_stopping_method="generate",
                # trim_intermediate_steps=5
            )
            
            # Execute the agent
            result = await agent_executor.ainvoke({"input": user_input})
            
            # Post-process the response to add HTML formatting while keeping natural flow
            response = result["output"]
            
            
            return response

        except Exception as e:
            return f"❌ An error occurred: {str(e)}"
        
    
    async def chat2(self, user_input: str) -> str:
        try:
            GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
            normalized_input = user_input.strip().lower()

            if normalized_input in GREETINGS:
                return (
                    "👋 Hi there! I’m Enriched AI — your personal travel planner. 🌍✨\n"
                    "Let me know which place you're dreaming of visiting, and I’ll help you with weather, top sights, and fun facts!"
                )
    


            async def chatgpt( user_input: str,category: str):
                """
                Generate general follow-up questions related to travel planning based on user input.
                Returns only the questions without location-specific names.
                """
                try:
                    

                    aclient = AsyncOpenAI()  # Add your actual API key here

                    client = AsyncIOMotorClient(
                        self.mongodb_link,
                        tls=True,
                        tlsCAFile=certifi.where()
                    )
                    db = client["followup_questions"]
                    collection_name = str(category)

                    collection = db.get_collection(collection_name)


                    response = await aclient.chat.completions.create(
                        model="gpt-4.1-nano",

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a travel assistant.\n"
                                    "Given a user's travel-related input, generate 2 follow-up questions "
                                    "that are generally useful for planning a trip.\n"
                                    "**Avoid including any specific locations or cities.**\n"
                                    "Respond ONLY with the questions, one per line. No numbering, no extra text, no greetings."
                                )
                            },
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    )

                    raw_output = response.choices[0].message.content.strip()
                    generated_questions = [q.strip().rstrip(".") + "" for q in raw_output.split("\n") if q.strip()]

                    # Insert into MongoDB
                    if generated_questions:
                        docs = [{"question": q} for q in generated_questions]
                        await collection.insert_many(docs)

                    return generated_questions

                except Exception as e:
                    print(f"❌ chatgpt error: {e}")
                    return []


            
            
            async def extract_query_category(user_input: str) -> Dict[str, Any]:
                """
                Return a JSON dict like:
                {"category": "places", "has_destination": true}

                Keys:
                category          – one of: weather | places | summary | travel | undefined
                has_destination   – true  if the text appears to include a city / country name
                                    false otherwise
                """
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            # We double‑brace {{ }} where we want a literal brace.
                            "You are a travel‑assistant *category extractor*.\n"
                            "Classify the user text into exactly one of:\n"
                            "- weather\n- places\n- summary\n- travel\n"
                            "Also decide if the text ALREADY names a destination "
                            "(a city, country or region).\n\n"
                            "✦ Respond **ONLY** with a JSON object in this form:\n"
                            "{{\"category\": \"<one_of_above>\", \"has_destination\": <true|false>}}\n"
                            "No extra words, no markdown, no explanations."
                        ),
                        ("user", "{input}")
                    ])
                    llm2 = ChatOllama(model="llama3.1:8b-instruct-q8_0",temperature=0)
                    chain   = prompt |  llm2          # your LLM wrapper
                    raw     = str(await chain.ainvoke({"input": user_input})).strip()

                    # The LLM should already reply with pure JSON.
                    match = re.search(r'\{[\s\S]*?\}', raw)
                    if match:
                        result = json.loads(match.group())
                        # Safety: ensure both keys are present
                        return {
                            "category":        result.get("category"),
                            "has_destination": bool(result.get("has_destination"))
                        }
                    raise ValueError("No JSON object found in LLM response.")

                except Exception as e:
                    print(f"LLM extraction error: {e}")
                    return {"category": None, "has_destination": False}

            async def search_followup_questions( user_input: str) -> str:
                try:

                    # 1. Extract category
                    category_result = await extract_query_category(user_input)
                    # print(f"\n\n\n\n\n categpry result : {category_result}\n\n\n\n\n\n")
                    category = category_result.get("category")

                    # 2. Connect to MongoDB
                    client = AsyncIOMotorClient(
                        self.mongodb_link,
                        tls=True,
                        tlsCAFile=certifi.where()
                    )
                    db = client["followup_questions"]
                    collection = db.get_collection(category)

                    # 3. Randomly fetch existing follow-up questions
                    results = await collection.aggregate([{"$sample": {"size": 2}}]).to_list(length=5)

                    # 4. Generate questions with ChatGPT if DB is empty
                    if not results:
                        print(f"\n \n Using Chatgpt\n \n")
                        generated_questions = await chatgpt(user_input,category)

                        if isinstance(generated_questions, list) and generated_questions:
                            docs = [{"question": q} for q in generated_questions]
                            await collection.insert_many(docs)

                            # Re-sample after insertion
                            results = await collection.aggregate([{"$sample": {"size": 2}}]).to_list(length=5)
                        else:
                            return f"1: {user_input.strip().rstrip('.')}?"

                    # 5. Format output
                    followups = [doc["question"].strip().rstrip(".") + "" for doc in results]
                    all_questions = [f"1: {user_input.strip().rstrip('.')}?"]
                    all_questions += [f"{i+2}: {q}" for i, q in enumerate(followups)]

                    return "\n".join(all_questions)

                except Exception as e:
                    print(f"❌ Error in search_followup_questions: {e}")
                    return f"1: {user_input.strip().rstrip('.')}?"

           
            user_input1 = await search_followup_questions(user_input=user_input)


            destination = await extract_query_category(user_input1)
            # print(f"\n\n Debug - Destination extraction result: {user_input1}\n \n \n \n ")


            response =  await self.chat_with_tools(user_input1)
           
           
            return response
        except Exception as e:
            return e
        


