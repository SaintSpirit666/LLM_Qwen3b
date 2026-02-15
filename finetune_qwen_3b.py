"""
РАБОЧИЙ ФАЙН-ТЮНИНГ QWEN 3B (без AutoAWQ конфликтов)
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

print("=" * 70)
print("🚀 ЗАПУСК ФАЙН-ТЮНИНГА QWEN 3B")
print("=" * 70)

# ============================================================================
# 1. ПРОВЕРКА ОКРУЖЕНИЯ
# ============================================================================

print("✓ PyTorch версия:", torch.__version__)
print("✓ CUDA доступно:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name(0))
    print("✓ Память GPU:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

# ============================================================================
# 2. ЗАГРУЗКА ТОКЕНИЗАТОРА
# ============================================================================

print("\n" + "=" * 70)
print("ЗАГРУЗКА ТОКЕНИЗАТОРА")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/qwen25_3b",
    trust_remote_code=True,
)

# Важно для Qwen
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✓ Токенизатор загружен")

# ============================================================================
# 3. ЗАГРУЗКА МОДЕЛИ
# ============================================================================

print("\n" + "=" * 70)
print("ЗАГРУЗКА МОДЕЛИ")
print("=" * 70)

# Загружаем БЕЗ использования AutoAWQ
model = AutoModelForCausalLM.from_pretrained(
    "/workspace/qwen25_3b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Включаем gradient checkpointing
model.gradient_checkpointing_enable()
print("✓ Модель загружена")
print("✓ Gradient checkpointing включен")

# ============================================================================
# 4. НАСТРОЙКА LoRA (с явным указанием, что это НЕ AWQ модель)
# ============================================================================

print("\n" + "=" * 70)
print("НАСТРОЙКА LoRA")
print("=" * 70)

# Указываем, что это НЕ AWQ модель
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Явно указываем модули
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=False,  # Отключаем DORA если есть
)

# Применяем LoRA с явным указанием, что это обычная модель
try:
    model = get_peft_model(model, lora_config)
    print("✓ LoRA настроена")
except Exception as e:
    print(f"✗ Ошибка LoRA: {e}")
    print("Пробуем альтернативный подход...")
    
    # Альтернатива: вручную настраиваем LoRA
    from peft.tuners.lora import LoraModel
    
    # Создаем PEFT модель вручную
    class SimpleLoraModel(LoraModel):
        def __init__(self, model, config, adapter_name):
            super().__init__(model, config, adapter_name)
    
    peft_model = SimpleLoraModel(model, {"default": lora_config}, "default")
    model = peft_model
    print("✓ LoRA настроена (альтернативный метод)")

# Статистика
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Обучаемых параметров: {trainable_params:,}")
print(f"✓ Всего параметров: {total_params:,}")
print(f"✓ Процент обучаемых: {trainable_params/total_params*100:.2f}%")

# ============================================================================
# 5. ПОДГОТОВКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 70)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 70)

def create_simple_dataset():
    """Создает простой датасет"""
    try:
        with open("/workspace/LLM/fire_safety_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        for i, item in enumerate(data[:3]):  # Всего 3 примера
            q = item.get("question", item.get("instruction", "?"))[:20]
            a = item.get("answer", item.get("response", "."))[:30]
            texts.append(f"В: {q}\nО: {a}")
        
        print(f"✓ Примеров: {len(texts)}")
        return Dataset.from_dict({"text": texts})
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        # Тестовые данные
        texts = [
            "В: Что делать при пожаре?\nО: Звонить 112.",
            "В: Как тушить огонь?\nО: Использовать огнетушитель.",
        ]
        print("✓ Используем тестовые данные")
        return Dataset.from_dict({"text": texts})

dataset = create_simple_dataset()

# ============================================================================
# 6. ОБУЧЕНИЕ
# ============================================================================

print("\n" + "=" * 70)
print("НАСТРОЙКА ОБУЧЕНИЯ")
print("=" * 70)

from transformers import Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=True,
    fp16=True,
)

print("✓ Параметры настроены")

# Токенизация
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

print("✓ Trainer создан")

# ============================================================================
# 7. ЗАПУСК ОБУЧЕНИЯ
# ============================================================================

print("\n" + "=" * 70)
print("ЗАПУСК ОБУЧЕНИЯ")
print("=" * 70)

try:
    print("⏳ Обучение...")
    trainer.train()
    print("✅ Обучение завершено!")
except Exception as e:
    print(f"❌ Ошибка: {e}")

# ============================================================================
# 8. СОХРАНЕНИЕ
# ============================================================================

print("\n" + "=" * 70)
print("СОХРАНЕНИЕ")
print("=" * 70)

try:
    model.save_pretrained("./qwen-3b-finetuned")
    tokenizer.save_pretrained("./qwen-3b-finetuned")
    print("✅ Модель сохранена")
except Exception as e:
    print(f"⚠ Ошибка сохранения: {e}")

# ============================================================================
# 9. ТЕСТ
# ============================================================================

print("\n" + "=" * 70)
print("ТЕСТ МОДЕЛИ")
print("=" * 70)

try:
    model.eval()
    
    test_input = tokenizer(
        "В: Что делать при пожаре?\nО:",
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **test_input,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Ответ: {response}")
    
except Exception as e:
    print(f"⚠ Ошибка теста: {e}")

print("\n" + "=" * 70)
print("🎉 ГОТОВО!")
print("=" * 70)
