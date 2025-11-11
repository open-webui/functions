# 🛡️ Moderation Filter

A lightweight **Open WebUI filter** that connects to a remote moderation service built with **LLM Guard**.
It checks user inputs before they are sent to the model and blocks unsafe or harmful content.

## 🚀 What It Does

* Sends each user message to a remote moderation API.
* The API scores the text for risk (e.g., toxicity, hate speech, unsafe content).
* If the content is unsafe, the filter blocks the message.
* Otherwise, it allows the prompt to proceed normally.

## ⚙️ Requirements

* Open WebUI (0.2.0 or newer)
* Python 3.10+
* `llm-guard`, `fastapi`, and `aiohttp` installed

You can find installation instructions for **LLM Guard** on the official site:
👉 [https://protectai.github.io/llm-guard](https://protectai.github.io/llm-guard)


## 🧩 How to Run

1. **Start the moderation API**

   * Run your FastAPI-based moderation service (the endpoint should accept a `POST` request at `/scan` and return whether a message is valid or not).
   * Keep this running on a reachable URL (for example: `http://localhost:8000/scan`).

2. **Add the filter to Open WebUI**

   * Add the `main.py` file in the WebUI admin panel, go to **Functions** and enable it.

3. **Configure the filter**

   * Set the `SCANNER_URL` valve to point to your moderation API endpoint.
   * Example: `http://localhost:8000/scan`

4. **Use as normal**

   * When users send prompts, the filter automatically checks them via the moderation API.
   * Unsafe inputs are blocked, and valid ones are passed through seamlessly.

## 🧠 Notes

* The filter runs **asynchronously** using `aiohttp` for better performance.
* If the moderation API is unreachable, the request is allowed (fail-open behavior).
* You can extend the same setup to also check **assistant outputs** if needed.

## ✅ Summary

Once running:

* Open WebUI sends each prompt to your moderation service.
* The service uses **LLM Guard** to score and decide.
* Unsafe content is stopped before it reaches the model.

That’s it — simple, fast, and effective moderation.
