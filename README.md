A simple Openrouter price tracker and balance checker tool for Open WebUI

Sometimes a bit clunky on smaller models, but generally it should work fine

Commands in chat (or use natural language):
- /price: Pulls current prices for all models added to the tool's respective Valve and marks changes in price
- /balance: Checks your current Openrouter balance (requires you to enter you Openrouter API key or environment varible "$OPENROUTER_API_KEY" in the Valve)
- /cheapest: lists the 10 currently cheapest models on Openrouter
- /price comparable (model-name): compares prices of similar models (fuzzy input tolerated)

Note: 
If you are running Open WebUi in a Docker container and prefer to not enter your Openrouter API key directly into the Valve you can pass an environment variable with your Openrouter API key in the docker run command:
docker run -e OPENROUTER_API_KEY="<your-api-key>"
Then enter $OPENROUTER_API_KEY into the API key valve of the tool.
