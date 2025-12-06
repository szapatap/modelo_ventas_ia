"""
Debug: Verificar estructura de BD y corregir save_prediction_db
Ejecuta esto en terminal para diagnosticar el problema
"""

import sqlite3
import pandas as pd

def debug_bd():
    """Función para depurar la estructura de la BD"""
    print("=" * 60)
    print("🔍 DIAGNÓSTICO DE BASE DE DATOS")
    print("=" * 60)
    
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    
    # Verificar si tabla exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    exists = c.fetchone()
    
    if not exists:
        print("❌ Tabla 'predictions' NO EXISTE")
        return
    
    print("✅ Tabla 'predictions' EXISTE")
    
    # Ver estructura
    print("\n📋 ESTRUCTURA DE COLUMNAS:")
    print("-" * 60)
    c.execute("PRAGMA table_info(predictions)")
    columns = c.fetchall()
    
    for col in columns:
        col_id, col_name, col_type, not_null, default, pk = col
        print(f"  • {col_name:20} | {col_type:10} | {'PK' if pk else ''}")
    
    # Ver datos
    print("\n📊 REGISTROS EN TABLA:")
    print("-" * 60)
    c.execute("SELECT COUNT(*) FROM predictions")
    count = c.fetchone()[0]
    print(f"  Total registros: {count}")
    
    if count > 0:
        c.execute("SELECT * FROM predictions LIMIT 3")
        rows = c.fetchall()
        print("\n  Últimos 3 registros:")
        for i, row in enumerate(rows, 1):
            print(f"    {i}. {row}")
    
    # Ver si hay tabla antigua
    print("\n🔄 TABLAS RELACIONADAS:")
    print("-" * 60)
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = c.fetchall()
    for table in all_tables:
        print(f"  • {table[0]}")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == '__main__':
    debug_bd()
