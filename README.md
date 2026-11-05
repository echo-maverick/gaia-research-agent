
# GAIA Research Agent

An open-source AI research agent built for the Hugging Face AI Agents Course Unit 4 final assessment.

## Overview

This project explores how an AI agent can combine language-model reasoning, web retrieval, audio transcription and structured data processing to answer complex questions.

It uses Qwen3-4B running locally, without requiring a paid inference API.

## Features

- Web search using DuckDuckGo
- Webpage reading and evidence extraction
- Local Qwen3-4B inference
- Audio transcription using OpenAI Whisper
- Excel spreadsheet analysis using openpyxl
- Automatic GAIA question retrieval
- Structured answer generation
- Basic error handling

## Technology

- Python
- Hugging Face Transformers
- Qwen3-4B
- OpenAI Whisper
- DuckDuckGo Search
- BeautifulSoup
- OpenPyXL
- Google Colab

## Project Structure

gaia_agent.py — Research agent and language model

file_tools.py — Audio and spreadsheet tools

evaluate.py — Assessment runner

requirements.txt — Dependencies

## Installation

Install dependencies:

    pip install -r requirements.txt

## Usage

Set your Hugging Face access token as an environment variable:

    export HF_TOKEN="your_token"

Run the assessment:

    python evaluate.py

The program retrieves GAIA questions, processes supported attachments, researches text questions and saves its results to gaia_agent_results.json.

## Limitations

This is an experimental research agent.

Current limitations include:
- Unreliable web-search availability
- Limited multimodal support
- Incomplete handling of images, video and executable code
- Possible factual and formatting errors

All outputs require evaluation before use.

## Course

Hugging Face AI Agents Course — Unit 4

https://huggingface.co/learn/agents-course/unit4/what-is-gaia

## Author

Hussain Razzaq

## Status

Experimental implementation. Official GAIA assessment score pending.
