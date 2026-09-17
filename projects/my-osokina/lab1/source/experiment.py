"""
Основной inference выполняется через OpenAI-compatible API Ollama:
    http://localhost:11434/v1/chat/completions

Для метрик используются usage из OpenAI-compatible ответа.
Latency — полное время HTTP-запроса.
output_tokens_per_sec — практическая скорость получения output:
output_tokens / latency_sec. 
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from prompts import PROMPTS


BASE_URL = "http://localhost:11434"
OPENAI_BASE_URL = f"{BASE_URL}/v1"
TIMEOUT = 600

MODELS: dict[str, str] = {
    "Phi-3.5-mini-3.8B-Q4_K_M": "phi3.5:3.8b-mini-instruct-q4_K_M",
    "Llama-3.2-3B": "llama3.2:3b",
    "Qwen2.5-3B": "qwen2.5:3b",
}

MODES: dict[str, dict[str, Any]] = {
    "A_baseline": {},

    "B_tuned": {
        "temperature": 0.30,
        "max_tokens": 150,
    },
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(
    base_url=f"{OPENAI_BASE_URL}/",
    api_key="ollama",
)


def warmup_model(model_id: str) -> None:
    """
    Один короткий warm-up перед основными измерениями.
    Он не включается в результаты.
    """
    client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "Ответь одним словом: готово"}
        ],
        temperature=0.0,
        max_tokens=8,
    )


def run_one(
    model_name: str,
    model_id: str,
    prompt_name: str,
    prompt_text: str,
    mode_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:

    print(
        f"\n[{model_name}] {prompt_name} | {mode_name}",
        flush=True,
    )

    started = time.perf_counter()

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        stream=False,
        **params,
    )

    elapsed_sec = time.perf_counter() - started

    answer = completion.choices[0].message.content or ""

    usage = completion.usage
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = (
        getattr(usage, "completion_tokens", None)
        if usage
        else None
    )
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    output_tokens_per_sec = None
    if output_tokens is not None and elapsed_sec > 0:
        output_tokens_per_sec = output_tokens / elapsed_sec

    result: dict[str, Any] = {
        "model_name": model_name,
        "model_id": model_id,
        "prompt": prompt_name,
        "mode": mode_name,
        "temperature": params.get("temperature"),
        "max_tokens": params.get("max_tokens"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": round(elapsed_sec, 4),
        "output_tokens_per_sec": (
            round(output_tokens_per_sec, 4)
            if output_tokens_per_sec is not None
            else None
        ),
        "response": answer,
    }

    print(f"  Input tokens:  {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Latency:       {elapsed_sec:.3f} sec")
    print(f"  Output tok/s:  {output_tokens_per_sec}")
    print("  Ответ:")
    print(answer)

    return result


def save_results(results: list[dict[str, Any]]) -> None:
    json_path = RESULTS_DIR / "raw_results.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    csv_path = RESULTS_DIR / "results.csv"

    fieldnames = [
        "model_name",
        "model_id",
        "prompt",
        "mode",
        "temperature",
        "max_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_sec",
        "output_tokens_per_sec",
        "response",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(
                {field: row.get(field) for field in fieldnames}
            )


def main() -> None:
    print("\nWarm-up:")
    for model_name, model_id in MODELS.items():
        print(f"  {model_name}", flush=True)
        warmup_model(model_id)

    total_runs = len(MODELS) * len(PROMPTS) * len(MODES)
    current_run = 0
    results: list[dict[str, Any]] = []

    print(f"\nОсновной эксперимент: {total_runs} запусков.")

    for model_name, model_id in MODELS.items():
        for prompt_name, prompt_text in PROMPTS.items():
            for mode_name, params in MODES.items():
                current_run += 1

                print(
                    f"\n--- Запуск {current_run}/{total_runs} ---",
                    flush=True,
                )

                try:
                    result = run_one(
                        model_name=model_name,
                        model_id=model_id,
                        prompt_name=prompt_name,
                        prompt_text=prompt_text,
                        mode_name=mode_name,
                        params=params,
                    )
                    results.append(result)
                    save_results(results)

                except Exception as exc:
                    print(
                        f"ОШИБКА: {exc}",
                        file=sys.stderr,
                    )
                    print(
                        "Промежуточные результаты уже сохранены."
                    )
                    save_results(results)
                    raise

    save_results(results)

    print()
    print("=" * 72)
    print("Эксперимент завершён.")
    print(f"Основных запусков: {len(results)}")
    print("Результаты:")
    print("  results/raw_results.json")
    print("  results/results.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
