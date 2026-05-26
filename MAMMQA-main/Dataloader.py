import os
import json
import pandas as pd
import base64
import csv
import io

# --- Helper Functions ---
def make_columns_unique(df):
    new_columns = []
    counts = {}
    for col in df.columns:
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

def encode_image(image_path, mime="jpeg"):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def csv_to_markdown(csv_string):
    f = io.StringIO(csv_string)
    reader = csv.reader(f)
    rows = list(reader)
    if not rows:
        return ""
    num_cols = max(len(row) for row in rows)
    for i, row in enumerate(rows):
        if len(row) < num_cols:
            rows[i] = row + [''] * (num_cols - len(row))
    col_widths = [max(len(cell.strip()) for cell in col) for col in zip(*rows)]
    header = "|" + "|".join(f" {rows[0][i].strip().ljust(col_widths[i])} " for i in range(num_cols)) + "|"
    divider = "|" + "|".join(":" + "-" * (col_widths[i] + 1) for i in range(num_cols)) + "|"
    data_rows = []
    for row in rows[1:]:
        data_rows.append("|" + "|".join(f" {row[i].strip().ljust(col_widths[i])} " for i in range(num_cols)) + "|")
    return "\n".join([header, divider] + data_rows)

# --- MultiModalQADataLoader (modified) ---
class MultiModalQADataLoader:
    def __init__(self, dev_file, tables_file, texts_file, images_file, 
                 images_base_url="path/to/local_images",  encode_images=False):
        self.dev_file = dev_file
        self.tables_file = tables_file
        self.texts_file = texts_file
        self.images_file = images_file
        self.images_base_url = images_base_url.rstrip("/")
        self.encode_images = encode_images

        self.dev_data = self.load_json(self.dev_file)
        self.tables = self.load_jsonl(self.tables_file)
        self.texts = self.load_jsonl(self.texts_file)
        self.images = self.load_jsonl(self.images_file)
        
        self.build_lookup()
        self.unified_data = self.build_unified_data()

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def load_jsonl(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping invalid line in {path}")
        return data

    def parse_table(self, data):
        title = data.get('title', 'No Title')
        data.pop('url')
        # url = data.get('url', '')
        table_block = data.get('table', {})
        headers = [h.get('column_name', '') for h in table_block.get('header', [])]
        rows = table_block.get('table_rows', [])
        formatted_rows = [[cell.get('text', '') for cell in row] for row in rows]
        
        if formatted_rows:
            if headers and len(headers) == len(formatted_rows[0]):
                df = pd.DataFrame(formatted_rows, columns=headers)
            else:
                df = pd.DataFrame(formatted_rows)
            df = make_columns_unique(df)
        else:
            df = pd.DataFrame()
        
        markdown_table = df.to_markdown(index=False)
        return {"title": title, "markdown_table": markdown_table}

    def parse_text(self, data):
        title = data.get('title', 'No Title')
        text = data.get('text', 'No Text')
        return {"title": title, "text": text}

    def parse_image(self, data):
        title = data.get('title', 'No Title')
        path = data.get('path', '')
        full_path = os.path.join(self.images_base_url, path)
        image_dict = {"title": title, "path": path, "url": full_path}
        
        if self.encode_images:
            image_dict["encoded_image"] = encode_image(full_path, "jpeg")
        return image_dict


    def build_lookup(self):
        self.table_lookup = { entry.get("id"): self.parse_table(entry)
                              for entry in self.tables if entry.get("id") }
        self.text_lookup = { entry.get("id"): self.parse_text(entry)
                             for entry in self.texts if entry.get("id") }
        self.image_lookup = { entry.get("id"): self.parse_image(entry)
                              for entry in self.images if entry.get("id") }

    def build_unified_data(self):
        unified = []
        for entry in self.dev_data:
            question = entry.get("question", "")
            answer = entry.get("answers", {})
            Type = entry["metadata"]["type"]
            Modality = entry["metadata"]["modalities"]
            metadata = entry.get("metadata", {})
            table_id = metadata.get("table_id")
            text_ids = metadata.get("text_doc_ids", [])
            image_ids = metadata.get("image_doc_ids", [])
            
            table_data = self.table_lookup.get(table_id)
            texts_data = [self.text_lookup[tid] for tid in text_ids if tid in self.text_lookup]
            images_data = [self.image_lookup[iid] for iid in image_ids if iid in self.image_lookup]
            
            unified.append({
                "question": question,
                "answer": answer,
                "type": Type,
                "modalities": Modality,
                "table": table_data,
                "texts": texts_data,
                "images": images_data
            })
        return unified

    def combine_texts(self, texts_list):
        return "\n".join(
            [f"title: {item.get('title', 'No Title')}, text: {item.get('text', 'No Text')}" for item in texts_list]
        )

    def get_agent_inputs(self, index):
        if index < 0 or index >= len(self.unified_data):
            return None
        
        example = self.unified_data[index]
        combined_text = self.combine_texts(example.get("texts", []))
        images = []
        for img in example.get("images", []):
            if self.encode_images and img.get("encoded_image"):
                images.append(img["encoded_image"])
            else:
                images.append(img.get("url"))
                
        table_md = ""
        if example.get("table"):
            # table_md = example["table"].get("markdown_table", "")
            table_md = example["table"]
                
        return {
            "question": example.get("question", ""),
            "type": example["type"],
            "modalities": example["modalities"],
            "text": combined_text,
            "table": table_md,
            "images": images,
            "answer": example.get("answer")
        }

# --- ManymodalQADataLoader (as is, with slight adjustments) ---
class ManymodalQADataLoader:
    def __init__(self, json_file, image_dir, encode_images=False):
        self.json_file = json_file
        self.image_dir = image_dir.rstrip("/")
        self.encode_images = encode_images

        self.dataframe = self.load_json_as_dataframe(self.json_file)
        self.questions_data = self.extract_questions_info(self.dataframe)

    def load_json_as_dataframe(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def extract_questions_info(self, df):
        extracted_data = []
        for _, row in df.iterrows():
            question_info = {
                "id": row['id'],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["q_type"],
                "text": row["text"] if pd.notna(row["text"]) else "No relevant text",
                "table": row["table"] if pd.notna(row["table"]) else "No table data",
                "image_id": row["id"],
                "caption": (row["image"]["caption"] if isinstance(row["image"], dict) and "caption" in row["image"]
                            else "No caption")
            }
            extracted_data.append(question_info)
        return extracted_data

    def get_image_path(self, image_id):
        
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            return image_path
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if os.path.exists(image_path):
            return image_path
        else:
            return "no_image"

    def get_agent_inputs(self, index, append_caption=True):
        if index < 0 or index >= len(self.questions_data):
            return None

        entry = self.questions_data[index]
        question = entry["question"]
        text = entry["text"]
        modality = entry["type"]
        table_data = entry["table"]
        caption = entry["caption"]
        answer = entry["answer"]

        if append_caption and caption != "No caption":
            text = f"{text} The name of the image is: {caption}"

        image_path = self.get_image_path(entry["image_id"])
        if self.encode_images and image_path != "no_image":
            image = encode_image(image_path, "png") or image_path
        else:
            image = image_path

        table_markdown = table_data
        if table_data != "No table data":
            table_markdown = csv_to_markdown(table_data)

        return {
            "question": question,
            "modalities":modality,
            "text": text,
            "table": table_markdown,
            "image": image,
            "id": entry["id"],
            "answer": answer
        }


class UnifiedQADataLoader:
    def __init__(self, dataset_type, captions_file=None, **kwargs):
        self.dataset_type = dataset_type.lower()
        self.captions = None
        if captions_file:
            with open(captions_file, "r") as f:
                self.captions = json.load(f)

        if self.dataset_type == "multimqa":
            self.loader = MultiModalQADataLoader(
                dev_file=kwargs.get("dev_file"),
                tables_file=kwargs.get("tables_file"),
                texts_file=kwargs.get("texts_file"),
                images_file=kwargs.get("images_file"),
                images_base_url=kwargs.get("images_base_url"),
                encode_images=kwargs.get("encode_images", False),
            )
        elif self.dataset_type == "manymqa":
            self.loader = ManymodalQADataLoader(
                json_file=kwargs.get("dev_file"),
                image_dir=kwargs.get("images_base_url"),
                encode_images=kwargs.get("encode_images", False)
            )
        else:
            raise ValueError("Unknown dataset type. Choose 'multimqa' or 'manymqa'.")

    def __len__(self):
        if self.dataset_type == "multimqa":
            return len(self.loader.unified_data)
        elif self.dataset_type == "manymqa":
            return len(self.loader.questions_data)

    def get_captions_for_images(self, image_paths):
        if not self.captions:
            return ""
        if self.dataset_type == "multimqa":
            return " ".join(
                "<caption><title>" + self.captions[os.path.basename(path).split('.')[0]]["title"] + "</title>" +
                self.captions[os.path.basename(path).split('.')[0]]["caption"] + "</caption>\n"
                for path in image_paths
                if os.path.basename(path).split('.')[0] in self.captions
            )
        elif self.dataset_type == "manymqa":
            captions = []
            for path in image_paths:
                key = os.path.basename(path).split('.')[0]
                if key in self.captions:
                    caption = self.captions[key].get("caption", "")
                    captions.append(f"<caption>{caption}</caption>\n")
                else:
                    captions.append("")
            return " ".join(captions)
        else:
            return ""

    def get_agent_inputs(self, index):
        data = self.loader.get_agent_inputs(index)
        if data is None:
            return None
        
        if self.dataset_type == "manymqa":
            # Convert singular 'image' key to 'images' list for consistency.
            data["images"] = [data.pop("image")]

        image_paths = data.get("images", [])
        data["captions"] = self.get_captions_for_images(image_paths)

        return data

    

if __name__ == "__main__":
    # Initialize the unified dataloader.
    # dataloader = UnifiedQADataLoader(
    #     dataset_type="multimqa",
    #     dev_file="./data/MultiModalQA/endgame_dev_filtered_data.json",
    #     tables_file="./data/MultiModalQA/MMQA_tables.jsonl",
    #     texts_file="./data/MultiModalQA/MMQA_texts.jsonl",
    #     images_file="./data/MultiModalQA/MMQA_images.jsonl",
    #     images_base_url="./data/MultiModalQA/final_dataset_images",
    #     captions_file="./data/MultiModalQA/MultiModelQA_Captions.json",
    #     encode_images=False
    # )

    dataloader = UnifiedQADataLoader(
        dataset_type="manymqa",
        dev_file="./data/ManyModalQA/ManyModalQAData/official_aaai_split_dev_data.json",
        tables_file="./data/MultiModalQA/MMQA_tables.jsonl",
        texts_file="./data/MultiModalQA/MMQA_texts.jsonl",
        images_file="./data/MultiModalQA/MMQA_images.jsonl",
        images_base_url="./data/ManyModalQA/ManyModalImages",
        captions_file="./data/ManyModalQA/ManyModelQA_Captions.json",
        encode_images=False
    )


    agent_inputs = dataloader.get_agent_inputs(100)
    print("text: ", agent_inputs["text"])
    print("Table:", agent_inputs["table"])
    print("modalitiy:", agent_inputs["modalities"])
    print("captions:", agent_inputs["captions"]) 
    # print(len(dataloader.loader.unified_data))
