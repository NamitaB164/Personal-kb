import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
import os
import re
from urllib.parse import urlparse

async def get_scrape_content(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url= url,
        )
        print(result.markdown)
        
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        path = parsed_url.path.strip("/").replace("/","_")
        raw_file_name = f"{hostname}_{path}" if path else hostname
        file = re.sub(r'[<>:"/\\|?*]', "_", raw_file_name) + ".md"
        
        folder_name = "data/scraped_files"
        file_path = os.path.join(folder_name,file)
        os.makedirs(folder_name, exist_ok=True)
        
        with open(file_path, "w") as md_file:
            md_file.write(result.markdown)
            
        return file_path


# asyncio.run(get_scrape_content(url))        

# if __name__ == "__main__":
#     asyncio.run(main())
