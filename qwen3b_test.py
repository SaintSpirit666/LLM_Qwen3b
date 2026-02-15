#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ТЕСТИРОВАНИЕ PEFT МОДЕЛИ ДЛЯ AMD ROCm 6.0
✅ БЕЗ BitsAndBytes (не работает с ROCm)
✅ Оптимизация памяти через gradient checkpointing
✅ CPU offload для экономии VRAM
✅ Merge модели для уменьшения памяти
"""

import os
import sys
import json
import torch
import gc
from pathlib import Path

# Настройка для AMD ROCm
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    import accelerate
except ImportError:
    print("❌ Установите: pip install transformers peft accelerate")
    sys.exit(1)


def print_header(text):
    """Красивый заголовок"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def check_gpu():
    """Проверка GPU и ROCm"""
    if not torch.cuda.is_available():
        print("❌ GPU не обнаружен!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {total_mem:.1f} GB")
    
    if "AMD" in gpu_name or "Radeon" in gpu_name:
        print("⚡ AMD GPU обнаружена - используем ROCm оптимизации")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    else:
        print("⚠️  Обнаружена NVIDIA GPU")
        return True


def get_vram_usage():
    """Текущее использование VRAM"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        return f"{allocated:.2f} GB / {total:.1f} GB (свободно: {free:.2f} GB)"
    return "GPU недоступен"


def clear_memory():
    """Агрессивная очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# МЕТОД 1: Merge на CPU, потом загрузка на GPU
# =============================================================================

def merge_on_cpu_save(base_model_path, peft_path, output_path):
    """
    ЛУЧШИЙ МЕТОД для AMD:
    1. Загружаем всё на CPU
    2. Объединяем LoRA с базовой моделью
    3. Сохраняем единую модель
    4. Очищаем память
    """
    print_header("МЕТОД 1: ОБЪЕДИНЕНИЕ НА CPU (РЕКОМЕНДУЕТСЯ)")
    
    print("📦 Загрузка базовой модели на CPU...")
    print("⏳ Это может занять 1-2 минуты...")
    
    # Загружаем на CPU с минимальным использованием памяти
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # float32 для CPU
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    print(f"✅ Базовая модель загружена на CPU")
    print(f"💻 RAM используется")
    
    print("\n📦 Загрузка PEFT адаптеров...")
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    
    print("✅ PEFT адаптеры загружены")
    
    print("\n🔧 Объединение LoRA с базовой моделью...")
    print("⏳ Это займет 30-60 секунд...")
    
    merged_model = peft_model.merge_and_unload()
    
    print("✅ Модели объединены!")
    
    print(f"\n💾 Сохранение объединенной модели в {output_path}...")
    
    # Создаем директорию
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Сохраняем модель
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"  # Разбиваем на части
    )
    
    # Сохраняем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Объединенная модель сохранена!")
    
    # Показываем размер файлов
    print("\n📁 Сохраненные файлы:")
    total_size = 0
    for file in Path(output_path).iterdir():
        if file.is_file():
            size = file.stat().st_size / 1024**2
            total_size += size
            print(f"  - {file.name}: {size:.1f} MB")
    print(f"\n📊 Общий размер: {total_size/1024:.2f} GB")
    
    # Очищаем память
    del base_model, peft_model, merged_model
    clear_memory()
    
    print("\n✅ Теперь можно загрузить модель на GPU!")
    return output_path


# =============================================================================
# МЕТОД 2: Загрузка с CPU offload (часть модели на CPU, часть на GPU)
# =============================================================================

def load_with_cpu_offload(model_path, offload_folder="./offload"):
    """
    Загрузка с CPU offload:
    - Часть слоев на GPU
    - Часть слоев на CPU
    - Автоматическое переключение
    """
    print_header("МЕТОД 2: CPU OFFLOAD")
    
    print("📦 Загрузка с автоматическим распределением...")
    print("⏳ Модель будет распределена между GPU и CPU")
    
    # Создаем папку для offload
    Path(offload_folder).mkdir(parents=True, exist_ok=True)
    
    # Загружаем с device_map="auto" - автоматически распределит
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Автоматическое распределение
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,  # Папка для временных данных
        offload_state_dict=True  # Разгружаем state dict
    )
    
    print("✅ Модель загружена с CPU offload")
    print(f"💾 VRAM: {get_vram_usage()}")
    
    return model


# =============================================================================
# МЕТОД 3: Последовательная загрузка (самый экономный)
# =============================================================================

def load_sequential(base_model_path, peft_path):
    """
    Последовательная загрузка:
    1. Загружаем базовую модель на GPU с float16
    2. Применяем PEFT адаптеры
    3. Используем gradient checkpointing для экономии
    """
    print_header("МЕТОД 3: ПОСЛЕДОВАТЕЛЬНАЯ ЗАГРУЗКА")
    
    clear_memory()
    
    print("📦 Загрузка базовой модели на GPU (float16)...")
    
    # Загружаем с минимальными требованиями к памяти
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "10GB"}  # Ограничиваем использование VRAM
    )
    
    print(f"✅ Базовая модель загружена")
    print(f"💾 VRAM: {get_vram_usage()}")
    
    # Включаем gradient checkpointing (экономит память)
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing включен (экономия памяти)")
    
    print("\n📦 Загрузка PEFT адаптеров...")
    
    try:
        model = PeftModel.from_pretrained(
            base_model,
            peft_path,
            is_trainable=False  # Только инференс
        )
        print(f"✅ PEFT модель загружена")
        print(f"💾 VRAM: {get_vram_usage()}")
        
        return model
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ Недостаточно памяти!")
            print("💡 Попробуйте:")
            print("   1. Метод 1 (Merge на CPU)")
            print("   2. Метод 2 (CPU offload)")
            clear_memory()
            return None
        raise


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_model(model, tokenizer, dataset_path, num_samples=3):
    """Тестирование модели на примерах"""
    print_header("ТЕСТИРОВАНИЕ МОДЕЛИ")
    
    # Проверяем устройство модели
    device = next(model.parameters()).device
    print(f"🎯 Модель на устройстве: {device}")
    print(f"💾 Текущий VRAM: {get_vram_usage()}")
    
    # Загружаем датасет
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Датасет: {len(data)} примеров")
    print(f"🧪 Тестируем на {num_samples} примерах\n")
    
    model.eval()  # Режим инференса
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'─'*70}")
        print(f"ПРИМЕР #{i+1}")
        print(f"{'─'*70}")
        
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")
        
        print(f"\n📝 Инструкция:\n{instruction}\n")
        print(f"✅ Ожидаемый ответ:\n{expected[:200]}...\n")
        
        # Формируем промпт (используем формат из обучения)
        prompt = f"Инструкция: {instruction}\n\nОтвет:"
        
        # Токенизация
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # На устройство модели
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Генерация
        print("🤖 Генерация ответа...")
        
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    renormalize_logits=True  # Предотвращение inf/nan
                )
            
            # Декодирование
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"🤖 Сгенерированный ответ:\n{generated}\n")
            print(f"💾 VRAM после генерации: {get_vram_usage()}")
            
            # Очистка после каждого примера
            del inputs, outputs
            clear_memory()
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ Недостаточно памяти для генерации!")
                print("💡 Попробуйте уменьшить max_new_tokens или использовать CPU offload")
                clear_memory()
                break
            raise


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    # Пути (измените на свои)
    BASE_MODEL_PATH = "/workspace/qwen25_3b"
    PEFT_PATH = "./qwen-3b-finetuned"
    MERGED_PATH = "./qwen-3b-merged"
    DATASET_PATH = "/workspace/LLM/fire_safety_dataset.json"
    
    print_header("ТЕСТИРОВАНИЕ PEFT МОДЕЛИ ДЛЯ AMD ROCm")
    
    # Проверка файлов
    if not Path(BASE_MODEL_PATH).exists():
        print(f"❌ Базовая модель не найдена: {BASE_MODEL_PATH}")
        return
    
    if not Path(PEFT_PATH).exists():
        print(f"❌ PEFT адаптеры не найдены: {PEFT_PATH}")
        return
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Датасет не найден: {DATASET_PATH}")
        return
    
    # Проверка GPU
    if not check_gpu():
        print("⚠️  GPU не обнаружен, работа будет на CPU (очень медленно)")
    
    print("\n📂 Файлы найдены:")
    print(f"  ✓ Базовая модель: {BASE_MODEL_PATH}")
    print(f"  ✓ PEFT адаптеры: {PEFT_PATH}")
    print(f"  ✓ Датасет: {DATASET_PATH}")
    
    # Выбор метода
    print("\n" + "="*70)
    print("ВЫБЕРИТЕ МЕТОД (для AMD ROCm 6.0):")
    print("="*70)
    print("1. Объединить модель на CPU и сохранить (РЕКОМЕНДУЕТСЯ)")
    print("   → Делается 1 раз, потом быстрая загрузка")
    print("   → Использует RAM, не GPU")
    print()
    print("2. Загрузить с CPU offload")
    print("   → Часть на GPU, часть на CPU")
    print("   → Экономит VRAM, но медленнее")
    print()
    print("3. Последовательная загрузка на GPU")
    print("   → Прямая загрузка, может не влезть в 12GB")
    print("   → Быстрее всего если влезет")
    print()
    print("4. Загрузить уже объединенную модель")
    print("   → Если уже сделали метод 1")
    print("="*70)
    
    choice = input("\nВаш выбор (1/2/3/4): ").strip()
    
    clear_memory()
    
    try:
        if choice == "1":
            # Объединение на CPU
            merged_path = merge_on_cpu_save(BASE_MODEL_PATH, PEFT_PATH, MERGED_PATH)
            
            print("\n✅ Модель объединена и сохранена!")
            print(f"📁 Путь: {merged_path}")
            print("\n💡 Теперь используйте метод 4 для тестирования")
            
            # Спрашиваем, загрузить ли сразу
            load_now = input("\nЗагрузить и протестировать сейчас? [y/N]: ").lower()
            if load_now == 'y':
                model = AutoModelForCausalLM.from_pretrained(
                    merged_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                tokenizer = AutoTokenizer.from_pretrained(merged_path)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                test_model(model, tokenizer, DATASET_PATH)
        
        elif choice == "2":
            # CPU offload - сначала нужно объединить
            if not Path(MERGED_PATH).exists():
                print("\n⚠️  Сначала нужно объединить модель (метод 1)")
                return
            
            model = load_with_cpu_offload(MERGED_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            test_model(model, tokenizer, DATASET_PATH)
        
        elif choice == "3":
            # Последовательная загрузка
            model = load_sequential(BASE_MODEL_PATH, PEFT_PATH)
            
            if model is None:
                print("\n❌ Не удалось загрузить модель")
                print("💡 Используйте метод 1 или 2")
                return
            
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            test_model(model, tokenizer, DATASET_PATH)
        
        elif choice == "4":
            # Загрузка объединенной
            if not Path(MERGED_PATH).exists():
                print(f"❌ Объединенная модель не найдена: {MERGED_PATH}")
                print("💡 Сначала выполните метод 1")
                return
            
            print_header("ЗАГРУЗКА ОБЪЕДИНЕННОЙ МОДЕЛИ")
            
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Модель загружена")
            print(f"💾 VRAM: {get_vram_usage()}")
            
            tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            test_model(model, tokenizer, DATASET_PATH)
        
        else:
            print("❌ Неверный выбор!")
            return
        
        print("\n" + "="*70)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("="*70)
        print(f"\n💾 Финальное использование VRAM: {get_vram_usage()}")
    
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        clear_memory()


if __name__ == "__main__":
    main()
