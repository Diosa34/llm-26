import os
import re
import datetime
import yaml
import pymupdf4llm
from bs4 import BeautifulSoup
import markdownify


def get_file_metadata(filepath):
    """Собирает базовые метаданные о файле."""
    stat = os.stat(filepath)
    date_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')

    return {
        'source': os.path.basename(filepath),
        'path': os.path.abspath(filepath),
        'date': date_str
    }


def save_markdown(content, metadata, output_path):
    """Сохраняет Markdown-файл с YAML-метаданными в начале (frontmatter)."""
    frontmatter = f"---\n{yaml.dump(metadata, allow_unicode=True, sort_keys=False)}---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter + content)
    print(f"Успешно сохранено: {output_path}")


def process_pdf(filepath, output_dir):
    """Обрабатывает PDF, извлекая текст постранично с сохранением структуры."""
    metadata = get_file_metadata(filepath)

    # Конвертируем PDF в Markdown с разбивкой по страницам
    md_pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)

    full_content = ""
    for i, page in enumerate(md_pages):
        # Безопасно извлекаем номер страницы.
        # Если в метаданных нет 'page', используем порядковый номер (i + 1)
        page_num = page.get('metadata', {}).get('page', i + 1)
        if not isinstance(page_num, int) or page_num < 1:
            page_num = i + 1

        # Безопасно извлекаем текст страницы
        text = page.get('text', '') if isinstance(page, dict) else str(page)

        # Добавляем маркер страницы для будущего чанкинга
        full_content += f"\n\n<!-- Страница {page_num} -->\n\n"
        full_content += text

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '.md'
    output_path = os.path.join(output_dir, output_filename)
    save_markdown(full_content, metadata, output_path)


def process_html(filepath, output_dir):
    """Обрабатывает HTML, извлекая только текст и сохраняя структуру заголовков/списков."""
    metadata = get_file_metadata(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Пытаемся вытащить заголовок страницы для метаданных раздела
    title_tag = soup.find('title')
    print(title_tag)
    if title_tag and title_tag.get_text(strip=True):
        metadata['section'] = title_tag.get_text(strip=True)
    else:
        # Если title нет, ищем первый h1
        h1_tag = soup.find('h1')
        metadata['section'] = h1_tag.get_text(strip=True) if h1_tag else 'Без заголовка'

    # Удаляем скрипты, стили и навигацию, чтобы остался только чистый контент
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    # Берем только тело документа для конвертации
    html_body = str(soup.body) if soup.body else str(soup)

    # Конвертируем в Markdown. heading_style="ATX" создает заголовки через #
    markdown_text = markdownify.markdownify(html_body, heading_style="ATX")

    # Убираем лишние пустые строки (3 и более переноса заменяем на 2)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '.md'
    output_path = os.path.join(output_dir, output_filename)
    save_markdown(markdown_text, metadata, output_path)


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Начинаю обработку файлов из папки: {input_dir}")

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.endswith("_files")]
        for file in files:
            filepath = os.path.join(root, file)
            ext = file.lower().split(".")[-1]

            if ext == "pdf":
                process_pdf(filepath, output_dir)
            elif ext in ["html", "htm"]:
                process_html(filepath, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HTML/PDF to Markdown with YAML-frontmatter")
    parser.add_argument(
        "--input",
        default=os.environ.get("CORPUS_DIR", r"C:\Users\Diosa\Desktop\Master\3\data2"),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("MARKDOWN_DIR", r"C:\Users\Diosa\Desktop\Master\3\markdown_for_llm_2"),
    )
    args = parser.parse_args()
    main(args.input, args.output)
