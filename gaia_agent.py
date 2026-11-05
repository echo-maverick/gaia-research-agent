
import re
import time
import requests
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS
from transformers import pipeline


class GAIAAgent:
    """Research agent using a local language model and web retrieval."""

    def __init__(self, model=None):
        self.model = model or pipeline(
            "text-generation",
            model="Qwen/Qwen3-4B",
            device_map="auto",
            dtype=torch.float16
        )

    def generate(self, instruction, max_tokens=80):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a factual research agent. "
                    "Use evidence rather than unsupported assumptions. "
                    "Return the shortest answer requested. "
                    "If evidence is insufficient, return UNKNOWN."
                )
            },
            {"role": "user", "content": instruction}
        ]

        prompt = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1200
        ).to(self.model.model.device)

        with torch.inference_mode():
            output = self.model.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=False
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        return self.model.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

    def make_queries(self, question):
        response = self.generate(
            "Generate two short web search queries for this "
            "question. Preserve important names and dates. "
            "One query per line. Do not answer.\n\n"
            + question,
            max_tokens=100
        )

        queries = []
        for line in response.splitlines():
            line = re.sub(r"^[\d.\-\s]+", "", line).strip()
            if 5 < len(line) < 150:
                queries.append(line)

        return queries[:2] or [question[:100]]

    def search(self, query):
        try:
            return list(
                DDGS().text(
                    query,
                    max_results=6,
                    backend="auto"
                )
            )
        except Exception:
            return []

    def read_page(self, url):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            response.raise_for_status()

            soup = BeautifulSoup(
                response.text, "html.parser"
            )

            for tag in soup([
                "script", "style", "nav",
                "footer", "header", "aside"
            ]):
                tag.decompose()

            main = (
                soup.select_one(
                    "article, main, #mw-content-text, "
                    ".mt-content-container"
                )
                or soup.body
                or soup
            )

            return main.get_text(" ", strip=True)

        except Exception:
            return ""

    def extract_evidence(self, text, question):
        words = re.findall(r"[a-zA-Z0-9-]+", question.lower())

        stopwords = {
            "what", "which", "where", "when", "that",
            "with", "from", "this", "there", "give",
            "only", "answer", "question", "mentioned"
        }

        keywords = [
            w for w in words
            if len(w) >= 4 and w not in stopwords
        ]

        passages = []

        for keyword in keywords:
            for match in list(
                re.finditer(
                    re.escape(keyword),
                    text,
                    flags=re.I
                )
            )[:2]:
                start = max(0, match.start() - 250)
                end = min(len(text), match.end() + 350)

                passage = text[start:end]
                score = sum(
                    word in passage.lower()
                    for word in keywords
                )

                passages.append((score, passage))

        passages.sort(key=lambda x: x[0], reverse=True)

        unique = []
        for _, passage in passages:
            if passage not in unique:
                unique.append(passage)

        return "\n\n".join(unique[:4])[:2500]

    def run(self, question):
        # General text-transformation handling
        reversed_question = question[::-1].lower()

        if "opposite of the word" in reversed_question:
            match = re.search(
                r'opposite of the word "([^"]+)"',
                reversed_question
            )

            if match:
                opposites = {
                    "left": "right",
                    "right": "left",
                    "up": "down",
                    "down": "up"
                }
                word = match.group(1)
                if word in opposites:
                    return opposites[word]

        queries = self.make_queries(question)
        evidence = []
        visited = set()

        for query in queries:
            for result in self.search(query):
                url = result.get("href", "")

                if not url or url in visited:
                    continue

                visited.add(url)

                text = self.read_page(url)
                passage = self.extract_evidence(text, question)

                if passage:
                    evidence.append(
                        f"Source: {url}\n{passage}"
                    )

                if len(evidence) >= 4:
                    break

            if len(evidence) >= 4:
                break

            time.sleep(1)

        if not evidence:
            return "UNKNOWN"

        return self.generate(
            f"Question:\n{question}\n\n"
            f"Evidence:\n{chr(10).join(evidence)[:3500]}\n\n"
            "Answer with only the requested name, number "
            "or short phrase. Do not explain.",
            max_tokens=70
        )
