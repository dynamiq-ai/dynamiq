# Agent Tool Calling with IBM Flow Engine Tools

## Prerequisitives

### Packages
* [Dynamiq](https://pypi.org/project/dynamiq/)
* [IBM Flows Engine](https://wxflows.ibm.stepzen.com/docs/installation)

### Api keys
* [OpenAI](https://openai.com/index/openai-api/)
* [IBM Flows Engine](https://wxflows.ibm.stepzen.com/docs/authentication)
* [OpenWeather](https://openweathermap.org/api)



### Setup API KEY for OpenWeather
Put OpenWeather api key in `.env` file inside `wxflows_endpoint` folder.
```
STEPZEN_OPEN_WEATHER_MAP_APIKEY=<openweather_api_key>
```

### Deploy tools
```
cd wxflows_endpoint
wxflows deploy
```

### Setup your API keys
Put your `OPENAI_KEY`, `FLOWS_ENGINE_API_KEY` and `URL` of endpoint (which you will get when run `deploy` command)
in `.env` in `wxflows` folder.

### Run python script
```
python ./agent.py
```


Endpoint was created with the following command:

```
wxflows init --endpoint-name api/my-endpoints \
    --import-name wikipedia \
    --import-package https://raw.githubusercontent.com/IBM/wxflows/refs/heads/main/tools/wikipedia.zip \
    --import-tool-name wikipedia \
    --import-tool-description "Retrieve information from Wikipedia." \
    --import-tool-fields "search|page" \
    --import-name weather  \
    --import-package https://raw.githubusercontent.com/IBM/wxflows/refs/heads/main/tools/weather.zip \
    --import-tool-name weather \
    --import-tool-description "Retrieve detailed weather information." \
    --import-tool-fields "weatherByCity"
```
