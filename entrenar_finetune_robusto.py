"""
Entrenador Robusto para Fine-tuning de CNN v2
"""
import signal
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ignorar señales de interrupción
def signal_handler(sig, frame):
    pass

signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

print("="*80)
print(" ENTRENADOR ROBUSTO - FINE-TUNING CNN v2")
print("="*80)
print("  Objetivo: Mejorar performance en canvas real")
print("  Dataset: 70% EMNIST + 30% Canvas sintético")
print("  Tiempo estimado: 20-30 minutos")
print("  Señales de interrupción IGNORADAS")
print("  Reintentos automáticos habilitados")
print("="*80)
print()

def entrenar_con_reintentos(max_reintentos=10):
    """Fine-tune CNN v2 con reintentos automáticos"""
    from src.finetune_cnn_v2 import fine_tune_cnn_v2
    
    for intento in range(1, max_reintentos + 1):
        try:
            print(f"\n{'='*80}")
            print(f"INTENTO {intento}/{max_reintentos}")
            print(f"{'='*80}\n")
            
            # Fine-tune
            model, history = fine_tune_cnn_v2(
                emnist_dir='data/raw',
                canvas_dir='data/canvas_synthetic',
                model_dir='models/cnn_modelo_v2',
                output_dir='models/cnn_modelo_v2_finetuned',
                epochs=30,
                batch_size=64,
                learning_rate=0.0001,
                canvas_weight=0.3,
                verbose=True
            )
            
            print(f"\n{'='*80}")
            print("  FINE-TUNING COMPLETADO EXITOSAMENTE")
            print(f"{'='*80}\n")
            return True
            
        except KeyboardInterrupt:
            print(f"\n{'!'*80}")
            print(f"   Interrupción detectada en intento {intento}")
            print(f"{'!'*80}")
            
            if intento < max_reintentos:
                print(f"  Reintentando en 5 segundos...")
                time.sleep(5)
                continue
            else:
                print(f"  Máximo de reintentos alcanzado ({max_reintentos})")
                return False
                
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"  ERROR INESPERADO en intento {intento}:")
            print(f"    {type(e).__name__}: {e}")
            print(f"{'!'*80}")
            
            if intento < max_reintentos:
                print(f"  Reintentando en 10 segundos...")
                time.sleep(10)
                continue
            else:
                print(f"  Máximo de reintentos alcanzado ({max_reintentos})")
                raise
    
    return False


if __name__ == '__main__':
    try:
        exito = entrenar_con_reintentos(max_reintentos=10)
        
        if exito:
            print("\n" + "="*80)
            print("  PROCESO COMPLETADO")
            print(" CNN v2 Fine-tuned está lista")
            print("  Mejorada para canvas real")
            print("  Ubicación: models/cnn_modelo_v2_finetuned/")
            print("="*80 + "\n")
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("   PROCESO INCOMPLETO")
            print(" El fine-tuning no pudo completarse")
            print("="*80 + "\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n{'='*80}")
        print(f" ❌ ERROR FATAL:")
        print(f"    {type(e).__name__}: {e}")
        print(f"{'='*80}\n")
        sys.exit(1)
