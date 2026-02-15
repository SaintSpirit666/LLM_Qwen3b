#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ЧАТ С ДООБУЧЕННОЙ QWEN 3B МОДЕЛЬЮ
✅ Streaming режим (живой вывод текста)
✅ AMD RX 7700 XT оптимизации
✅ Сохранение/загрузка диалогов
✅ Настройка параметров генерации
"""

import os
import sys
import io
import json
import torch
import signal
from pathlib import Path
from datetime import datetime
from threading import Thread

# UTF-8 настройка
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# AMD ROCm настройки
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("❌ Установите: pip install transformers torch")
    sys.exit(1)


# =============================================================================
# Модель
# =============================================================================

class FinetunedQwenChat:
    """Чат с дообученной моделью"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.history = []
        self.stats = {
            "start": datetime.now(),
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0
        }
        
        print("🔧 Настройка окружения...")
        self._setup_env()
        
        print(f"📂 Загрузка модели из: {model_path}")
        self._load_model()
        
        # Системный промпт (адаптируйте под вашу задачу)
        self.system_prompt = """Ты — специализированный ассистент, обученный отвечать на вопросы по пожарной безопасности.

ПРАВИЛА:
- Отвечай ТОЛЬКО на русском языке
- Давай точные, профессиональные ответы
- Ссылайся на нормативные документы если знаешь
- Если не уверен - скажи честно"""
        
        print(f"✅ Модель загружена на: {self.device}")
        print(f"💾 VRAM: {self._get_vram()}\n")
    
    def _setup_env(self):
        """Настройка окружения"""
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU: {name}")
            
            if "AMD" in name or "Radeon" in name:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("⚡ ROCm оптимизации включены")
    
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        # Токенизатор
        print("🔤 Загрузка токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Устройство: {device.upper()}")
        
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Модель
        print("🧠 Загрузка модели (может занять 1-2 минуты)...")
        
        if device == "cuda":
            try:
                # Пробуем сначала загрузить на GPU с ограничением
                print("💡 Пробую загрузить на GPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "9GB", "cpu": "16GB"}
                )
                print("✅ Успешно загружено на GPU")
                
            except torch.cuda.OutOfMemoryError:
                print("⚠️  Недостаточно памяти, загружаю на CPU...")
                # Загружаем на CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # fp32 для CPU
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                # Переносим часть на GPU
                print("⚡ Переношу часть модели на GPU...")
                self.model = self.model.to(dtype)
                # Только первые слои на GPU
                for name, module in self.model.named_children():
                    if name in ["model", "lm_head"]:
                        module = module.to("cuda:0")
                
        else:
            # CPU only
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"✅ Модель загружена на: {self.device}")
    
    def _get_vram(self) -> str:
        """Информация о VRAM"""
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{used:.1f}/{total:.1f} GB ({used/total*100:.0f}%)"
        return "CPU"
    
    def _build_prompt(self, user_message: str) -> str:
        """Построение промпта в формате обучения"""
        # Используем тот же формат, что и при обучении
        return f"Инструкция: {user_message}\n\nОтвет:"
    
    def _build_chat_prompt(self) -> str:
        """Построение промпта с историей для чата"""
        lines = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>"]
        
        for msg in self.history:
            role, content = msg["role"], msg["content"]
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)
    
    def chat(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """Генерация ответа"""
        
        # Добавляем в историю
        self.history.append({"role": "user", "content": user_message})
        
        # Обрезка истории
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        # Промпт (можете выбрать один из двух форматов)
        # Вариант 1: Простой формат из обучения
        # prompt = self._build_prompt(user_message)
        
        # Вариант 2: Чат формат с историей
        prompt = self._build_chat_prompt()
        
        # Токенизация
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tokens,
            add_special_tokens=False
        )
        
        if self.device.type in ["cuda", "hip"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[1]
        
        # Streaming
        if stream:
            return self._stream_generate(inputs, input_len, max_tokens, temperature, top_p)
        
        # Обычная генерация
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.1, min(temperature, 2.0)),
                top_p=top_p,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                renormalize_logits=True
            )
        
        # Декодирование
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        response = self._clean_response(response)
        
        # Сохраняем в историю
        self.history.append({"role": "assistant", "content": response})
        
        # Статистика
        self.stats["requests"] += 1
        self.stats["tokens_in"] += input_len
        self.stats["tokens_out"] += len(outputs[0]) - input_len
        
        return response
    
    def _stream_generate(self, inputs, input_len, max_tokens, temperature, top_p):
        """Потоковая генерация с живым выводом"""
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(0.1, min(temperature, 2.0)),
            "top_p": top_p,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "no_repeat_ngram_size": 3,
            "renormalize_logits": True,
            "streamer": streamer
        }
        
        # Запуск в отдельном потоке
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Живой вывод
        full_response = []
        try:
            for token in streamer:
                if token:
                    print(token, end='', flush=True)
                    full_response.append(token)
        except KeyboardInterrupt:
            print("\n⚠️  Генерация прервана")
        finally:
            thread.join(timeout=30)
        
        response = ''.join(full_response)
        response = self._clean_response(response)
        
        # Сохраняем в историю
        if response:
            self.history.append({"role": "assistant", "content": response})
        
        self.stats["requests"] += 1
        self.stats["tokens_in"] += input_len
        
        return response
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа от служебных токенов"""
        import re
        
        # Убираем токены Qwen
        patterns = [
            r'<\|im_start\|>\s*assistant\s*\n?',
            r'<\|im_start\|>\s*user\s*\n?',
            r'<\|im_start\|>\s*system\s*\n?',
            r'<\|im_end\|>',
            r'<\|endoftext\|>',
        ]
        for p in patterns:
            text = re.sub(p, '', text)
        
        # Нормализация пробелов
        text = ' '.join(text.split())
        
        return text.strip()
    
    def reset(self):
        """Очистка истории"""
        self.history.clear()
        print("🧹 История очищена")
    
    def save_history(self, path: str = None):
        """Сохранение диалога"""
        if not path:
            path = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "time": datetime.now().isoformat(),
            "model": self.model_path,
            "history": self.history,
            "stats": {
                "requests": self.stats["requests"],
                "tokens": self.stats["tokens_in"] + self.stats["tokens_out"]
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Сохранено: {path}")
    
    def load_history(self, path: str):
        """Загрузка диалога"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.history = data.get("history", [])
        print(f"✅ Загружено {len(self.history)} сообщений")
    
    def get_stats(self):
        """Статистика"""
        uptime = datetime.now() - self.stats["start"]
        return {
            "время": str(uptime).split('.')[0],
            "запросов": self.stats["requests"],
            "токенов_всего": self.stats["tokens_in"] + self.stats["tokens_out"],
            "vram": self._get_vram()
        }


# =============================================================================
# CLI интерфейс
# =============================================================================

class ChatCLI:
    """Интерфейс командной строки"""
    
    def __init__(self, model_path: str):
        self.bot = FinetunedQwenChat(model_path)
        self.stream_mode = True  # По умолчанию streaming
        self.max_tokens = 512
        self.temperature = 0.7
        self.running = True
        
        signal.signal(signal.SIGINT, self._interrupt)
    
    def _interrupt(self, sig, frame):
        print("\n\n🛑 Прервано")
        self.running = False
    
    def _read_line(self) -> str:
        """Чтение строки"""
        try:
            return sys.stdin.readline().rstrip('\n\r')
        except:
            return ""
    
    def print_help(self):
        """Справка"""
        print("""
╔════════════════════════════════════════════════════════════╗
║                    КОМАНДЫ ЧАТА                            ║
╠════════════════════════════════════════════════════════════╣
║  /help        - эта справка                                ║
║  /exit        - выход                                      ║
║  /clear       - очистить историю                           ║
║  /save [file] - сохранить диалог                          ║
║  /load <file> - загрузить диалог                          ║
║  /stats       - статистика                                 ║
║  /stream      - переключить streaming (сейчас: {})         ║
║  /tokens N    - макс. токенов (50-2000)                   ║
║  /temp X      - temperature (0.1-2.0)                     ║
║  /vram        - показать использование памяти              ║
╚════════════════════════════════════════════════════════════╝
        """.format("ON" if self.stream_mode else "OFF"))
    
    def run(self):
        """Главный цикл"""
        print("\n" + "="*70)
        print("🤖 ЧАТ С ДООБУЧЕННОЙ QWEN 3B МОДЕЛЬЮ")
        print("="*70)
        print("💡 /help - справка | /exit - выход")
        print(f"🌊 Streaming: {'ВКЛЮЧЕН' if self.stream_mode else 'ВЫКЛЮЧЕН'}")
        print("="*70 + "\n")
        
        while self.running:
            try:
                print("👤 > ", end='', flush=True)
                text = self._read_line()
                
                if not text:
                    continue
                
                # Команды
                if text.startswith('/'):
                    if self._handle_command(text):
                        break
                    continue
                
                # Генерация
                print("🤖 ", end='', flush=True)
                
                response = self.bot.chat(
                    text,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=self.stream_mode
                )
                
                if not self.stream_mode:
                    print(response)
                
                print()  # Новая строка
                
            except KeyboardInterrupt:
                print("\n\n🛑 Для выхода: /exit")
                continue
            except Exception as e:
                print(f"\n⚠️  Ошибка: {e}\n")
        
        # Финал
        print("\n" + "="*70)
        print("📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        for k, v in self.bot.get_stats().items():
            print(f"  {k}: {v}")
        print("="*70)
        print("👋 До свидания!\n")
    
    def _handle_command(self, cmd: str) -> bool:
        """Обработка команд. True = выход"""
        parts = cmd.split()
        c = parts[0].lower()
        
        if c == "/exit":
            # Предложить сохранить
            if len(self.bot.history) > 0:
                save = input("\n💾 Сохранить диалог? [y/N]: ").lower()
                if save == 'y':
                    self.bot.save_history()
            return True
        
        elif c == "/help":
            self.print_help()
        
        elif c == "/clear":
            self.bot.reset()
        
        elif c == "/save":
            file = parts[1] if len(parts) > 1 else None
            self.bot.save_history(file)
        
        elif c == "/load" and len(parts) > 1:
            self.bot.load_history(parts[1])
        
        elif c == "/stats":
            print("\n📊 СТАТИСТИКА:")
            for k, v in self.bot.get_stats().items():
                print(f"  {k}: {v}")
            print()
        
        elif c == "/stream":
            self.stream_mode = not self.stream_mode
            status = "ВКЛЮЧЕН ✓" if self.stream_mode else "ВЫКЛЮЧЕН ✗"
            print(f"🌊 Streaming: {status}")
        
        elif c == "/tokens" and len(parts) > 1:
            try:
                n = int(parts[1])
                if 50 <= n <= 2000:
                    self.max_tokens = n
                    print(f"✅ Max tokens: {n}")
                else:
                    print("❌ Диапазон: 50-2000")
            except:
                print("❌ Формат: /tokens 512")
        
        elif c == "/temp" and len(parts) > 1:
            try:
                t = float(parts[1])
                if 0.1 <= t <= 2.0:
                    self.temperature = t
                    print(f"✅ Temperature: {t}")
                else:
                    print("❌ Диапазон: 0.1-2.0")
            except:
                print("❌ Формат: /temp 0.7")
        
        elif c == "/vram":
            print(f"💾 {self.bot._get_vram()}")
        
        else:
            print(f"❌ Неизвестная команда: {c}")
            print("💡 /help - список команд")
        
        return False


# =============================================================================
# Точка входа
# =============================================================================

def main():
    print("╔" + "="*68 + "╗")
    print("║      ЧАТ С ДООБУЧЕННОЙ МОДЕЛЬЮ - QWEN 3B (MERGED)         ║")
    print("║             AMD RX 7700 XT + ROCm 6.0                     ║")
    print("╚" + "="*68 + "╝\n")
    
    # Путь к объединенной модели
    default_path = "./qwen-3b-merged"
    
    print(f"📂 Путь к модели [{default_path}]: ", end='', flush=True)
    path = sys.stdin.readline().rstrip('\n\r')
    
    if not path:
        path = default_path
    
    # Проверка
    if not Path(path).exists():
        print(f"\n❌ Модель не найдена: {path}")
        print("\n💡 Убедитесь что вы:")
        print("   1. Дообучили модель")
        print("   2. Объединили её (test_peft_model_amd.py → метод 1)")
        print("   3. Указали правильный путь")
        return 1
    
    try:
        cli = ChatCLI(path)
        cli.run()
        return 0
    
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
