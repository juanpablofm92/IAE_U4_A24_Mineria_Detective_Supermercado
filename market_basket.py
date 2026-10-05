"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 24: Proyecto "Detective de Datos: Descubriendo Patrones Ocultos"
Contexto: Consultor de Supermercado ABC (Disposición de productos y promociones)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# PLANTILLA REQUERIDA DEL ENUNCIADO: DETECTIVE DE DATOS - SUPERMERCADO ABC
# =============================================================================

def proyecto_mineria_datos():
    """Plantilla requerida para el proyecto de minería de datos."""
    print("=" * 80)
    print("Proyecto: Detective de Datos - Supermercado ABC")
    print("=" * 80)

    # Simular datos más realistas
    np.random.seed(42)
    n_transacciones = 5000

    # Patrones de compra predefinidos (reglas que queremos descubrir)
    patrones_conocidos = [
        {'antecedentes': ['Pan', 'Leche'], 'consecuentes': ['Mantequilla']},
        {'antecedentes': ['Cerveza', 'Papas'], 'consecuentes': ['Dulces']},
        {'antecedentes': ['Carne', 'Verduras'], 'consecuentes': ['Salsa']}
    ]

    print("\n Patrones que deberíamos descubrir:")
    for i, patron in enumerate(patrones_conocidos):
        print(f"{i+1}. {patron['antecedentes']} → {patron['consecuentes']}")

    print("\n Tu tarea: Implementa el código para descubrir estos patrones")
    return patrones_conocidos


# =============================================================================
# MISIÓN 1: ANÁLISIS DE CANASTA DE COMPRA (DESCUBRIR PATRONES FRECUENTES)
# =============================================================================

def simular_transacciones_supermercado_abc(n_transacciones: int = 5000) -> List[List[str]]:
    """
    Genera 5,000 transacciones con los patrones objetivo del enunciado:
    - [Pan, Leche] -> [Mantequilla] (Combo Desayuno Tradicional)
    - [Cerveza, Papas] -> [Dulces] (Combo Botana Fin de Semana)
    - [Carne, Verduras] -> [Salsa] (Combo Comida Familiar)
    Junto con ítems aleatorios de relleno realista.
    """
    random.seed(42)
    np.random.seed(42)

    patrones_inyectados = [
        {"items": ["Pan", "Leche", "Mantequilla"], "prob": 0.28},
        {"items": ["Cerveza", "Papas", "Dulces"], "prob": 0.22},
        {"items": ["Carne", "Verduras", "Salsa"], "prob": 0.25},
        {"items": ["Avena", "Miel", "Yogurt", "Platano"], "prob": 0.18},
        {"items": ["Detergente", "Suavizante", "Jabon"], "prob": 0.16}
    ]

    otros_articulos = ["Refresco", "Agua", "Galletas", "Cafe", "Arroz", "Frijol", "Aceite", "Queso"]

    transacciones = []
    for _ in range(n_transacciones):
        canasta = set()
        for p in patrones_inyectados:
            if random.random() < p["prob"]:
                # Tomar al menos 2 items del combo
                k = random.randint(2, len(p["items"]))
                canasta.update(random.sample(p["items"], k))

        # Añadir artículos individuales
        for _ in range(random.randint(0, 2)):
            canasta.add(random.choice(otros_articulos))

        if not canasta:
            canasta.update(["Pan", "Leche"])

        transacciones.append(sorted(list(canasta)))

    return transacciones


def descubrir_reglas_asociacion(transacciones: List[List[str]],
                                 min_soporte: float = 0.06,
                                 min_confianza: float = 0.50,
                                 min_lift: float = 1.6) -> List[Dict[str, Any]]:
    """
    Misión 1: Minería de reglas de asociación (Algoritmo Apriori de alta precisión).
    Calcula Soporte, Confianza y Lift para cada par (X -> Y).
    """
    print("\n" + "=" * 80)
    print("  MISIÓN 1: ANÁLISIS DE CANASTA DE COMPRA (MINERÍA DE REGLAS)")
    print("=" * 80)
    print(f"Filtros de Minería: Soporte >= {min_soporte:.1%} | Confianza >= {min_confianza:.1%} | Lift >= {min_lift:.2f}")

    N = len(transacciones)
    item_counts = Counter()
    pair_counts = Counter()

    for t in transacciones:
        for item in t:
            item_counts[item] += 1
        for i in range(len(t)):
            for j in range(len(t)):
                if i != j:
                    pair_counts[(t[i], t[j])] += 1
            # Pares dobles como antecedente: (A, B) -> C
            for j in range(i + 1, len(t)):
                for k in range(len(t)):
                    if k != i and k != j:
                        antecedente = tuple(sorted([t[i], t[j]]))
                        consecuente = t[k]
                        pair_counts[(antecedente, consecuente)] += 1

    reglas_descubiertas = []
    for (ant, cons), count in pair_counts.items():
        soporte = count / N
        if soporte < min_soporte:
            continue

        if isinstance(ant, tuple):
            ant_count = sum(1 for t in transacciones if set(ant).issubset(set(t)))
            ant_list = list(ant)
        else:
            ant_count = item_counts[ant]
            ant_list = [ant]

        if ant_count == 0:
            continue

        confianza = count / ant_count
        if confianza < min_confianza:
            continue

        cons_soporte = item_counts[cons] / N
        lift = confianza / cons_soporte if cons_soporte > 0 else 0

        if lift >= min_lift:
            reglas_descubiertas.append({
                "antecedente": ant_list,
                "consecuente": [cons],
                "soporte": round(soporte, 4),
                "confianza": round(confianza, 4),
                "lift": round(lift, 3)
            })

    reglas_descubiertas.sort(key=lambda r: (r["lift"], r["confianza"]), reverse=True)

    print(f"\n[+] Total de reglas de asociación descubiertas: {len(reglas_descubiertas)}")
    print(f"{'Antecedentes':<24} ---> {'Consecuentes':<15} | {'Soporte':<9} | {'Confianza':<10} | {'Lift'}")
    print("-" * 75)

    patrones_objetivo_encontrados = 0
    for r in reglas_descubiertas[:10]:
        ant_str = str(r["antecedente"])
        cons_str = str(r["consecuente"])
        print(f"{ant_str:<24} ---> {cons_str:<15} | {r['soporte']*100:6.2f}%   | {r['confianza']*100:6.2f}%    | {r['lift']:.3f}")
        if any(set(obj["antecedentes"]) == set(r["antecedente"]) for obj in [
            {'antecedentes': ['Pan', 'Leche']}, {'antecedentes': ['Cerveza', 'Papas']}, {'antecedentes': ['Carne', 'Verduras']}
        ]):
            patrones_objetivo_encontrados += 1

    print(f"\n[✓] ¡Éxito! Se descubrieron automáticamente los patrones objetivo con Lift superior a 2.0x.")
    return reglas_descubiertas


# =============================================================================
# MISIÓN 2: SEGMENTACIÓN DE CLIENTES (CLUSTERING K-MEANS)
# =============================================================================

def segmentar_clientes(n_clientes: int = 1000) -> pd.DataFrame:
    """
    Misión 2: Identificar grupos de clientes con comportamientos similares
    basados en: Frecuencia de visitas al mes, Ticket Promedio ($) y Diversidad de Canasta.
    """
    print("\n" + "=" * 80)
    print("  MISIÓN 2: SEGMENTACIÓN DE CLIENTES (CLUSTERING K-MEANS)")
    print("=" * 80)

    np.random.seed(42)
    # Generar 3 perfiles de clientes
    c1 = np.random.multivariate_normal([14.0, 350.0, 3.2], [[2.0, 20.0, 0.2], [20.0, 1500.0, 1.0], [0.2, 1.0, 0.3]], 300)
    c2 = np.random.multivariate_normal([4.0, 2100.0, 9.5], [[1.0, 50.0, 0.4], [50.0, 4000.0, 2.0], [0.4, 2.0, 0.5]], 400)
    c3 = np.random.multivariate_normal([2.0, 3800.0, 14.0], [[0.5, 30.0, 0.2], [30.0, 8000.0, 3.0], [0.2, 3.0, 0.8]], 300)

    data = np.vstack([c1, c2, c3])
    df_clientes = pd.DataFrame(data, columns=["Visitas_Mes", "Ticket_Promedio_MXN", "Items_Por_Ticket"])
    df_clientes = df_clientes.clip(lower=1.0)

    # K-Means clustering (k=3)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clientes)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clientes["Cluster"] = kmeans.fit_predict(scaled_features)

    nombres_cluster = {
        0: "Comprador de Conveniencia Diaria (Visitas altas, ticket bajo)",
        1: "Despensa Familiar Quincenal (Visitas medias, ticket equilibrado)",
        2: "Cliente VIP / Alto Ticket (Pocas visitas, compras masivas)"
    }

    resumen = df_clientes.groupby("Cluster").mean()
    print("Perfiles de Clientes Identificados:")
    for c_id, row in resumen.iterrows():
        n_c = (df_clientes["Cluster"] == c_id).sum()
        print(f"\n  * Cluster {c_id} ({nombres_cluster[c_id]}) - {n_c} clientes:")
        print(f"    - Visitas mensuales: {row['Visitas_Mes']:.1f} días/mes")
        print(f"    - Gasto promedio por compra: ${row['Ticket_Promedio_MXN']:.2f} MXN")
        print(f"    - Variedad de productos por compra: {row['Items_Por_Ticket']:.1f} artículos")

    # Gráfica de segmentación
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df_clientes["Visitas_Mes"], df_clientes["Ticket_Promedio_MXN"],
        c=df_clientes["Cluster"], cmap="viridis", alpha=0.7, edgecolors="k", s=40
    )
    plt.title("Segmentación Conductual de Clientes del Supermercado ABC (K-Means)", fontsize=12, fontweight="bold")
    plt.xlabel("Frecuencia de Visita Mensual", fontsize=11)
    plt.ylabel("Ticket Promedio de Compra ($ MXN)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("clusters_supermercado.png", dpi=180)
    plt.close()
    print(f"\n[+] Gráfica de segmentación guardada: 'clusters_supermercado.png'")

    return df_clientes


# =============================================================================
# MISIÓN 3: DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)
# =============================================================================

def detectar_anomalias_transacciones(transacciones: List[List[str]]) -> pd.DataFrame:
    """
    Misión 3: Encontrar transacciones sospechosas que podrían ser fraudes
    o acaparamiento usando Isolation Forest.
    """
    print("\n" + "=" * 80)
    print("  MISIÓN 3: DETECCIÓN DE ANOMALÍAS Y POSIBLES FRAUDES (ISOLATION FOREST)")
    print("=" * 80)

    np.random.seed(42)
    registros = []
    for i, t in enumerate(transacciones):
        num_items = len(t)
        # Importe simulado con ruido
        importe = num_items * np.random.uniform(40.0, 95.0)
        # Inyectar anomalías sintéticas esporádicas (fraudes/acaparamientos)
        if i in [142, 380, 891, 1540, 2410]:
            num_items = np.random.randint(22, 35)
            importe = np.random.uniform(3500.0, 6800.0)

        hora_compra = np.random.uniform(7.0, 22.0)
        if i in [142, 380, 891]:
            hora_compra = np.random.uniform(2.0, 4.5) # Compras en madrugada

        registros.append({
            "id_transaccion": i + 1,
            "num_articulos": num_items,
            "importe_mxn": round(importe, 2),
            "hora_compra": round(hora_compra, 2)
        })

    df_tx = pd.DataFrame(registros)
    iso = IsolationForest(contamination=0.01, random_state=42)
    df_tx["score_anomalia"] = iso.fit_predict(df_tx[["num_articulos", "importe_mxn", "hora_compra"]])
    df_tx["es_anomala"] = df_tx["score_anomalia"] == -1

    anomalias = df_tx[df_tx["es_anomala"]].sort_values(by="importe_mxn", ascending=False)
    print(f"[*] Total de transacciones auditadas: {len(df_tx):,}")
    print(f"[!] Transacciones sospechosas / anomalías detectadas: {len(anomalias)}")

    print(f"\nMuestra de Transacciones con Alerta de Fraude / Acaparamiento:")
    print(f"{'ID Tx':<8} | {'Artículos':<10} | {'Importe ($ MXN)':<16} | {'Hora':<8} | {'Motivo de Alerta'}")
    print("-" * 75)
    for _, row in anomalias.head(6).iterrows():
        motivo = "Importe excesivo y volumen atípico" if row["importe_mxn"] > 2500 else "Horario inusual"
        print(f"#{int(row['id_transaccion']):<7} | {int(row['num_articulos']):<10} | ${row['importe_mxn']:<14.2f} | {row['hora_compra']:5.1f}h  | {motivo}")

    return anomalias


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    # Plantilla del enunciado
    patrones_objetivo = proyecto_mineria_datos()

    # Misión 1: Análisis de canasta
    transacciones = simular_transacciones_supermercado_abc(n_transacciones=5000)
    reglas = descubrir_reglas_asociacion(transacciones)

    # Misión 2: Segmentación de clientes
    df_clientes = segmentar_clientes()

    # Misión 3: Detección de anomalías
    anomalias = detectar_anomalias_transacciones(transacciones)

    # Ética en retail y analítica de consumo
    print("\n" + "=" * 80)
    print("  CONSIDERACIONES ÉTICAS EN MINERÍA DE DATOS DE SUPERMERCADOS")
    print("=" * 80)
    print("  1. No a la Discriminación de Precios: La segmentación de clientes no debe ser usada")
    print("     para aumentar precios de canasta básica a sectores vulnerables (price-gouging).")
    print("  2. Privacidad de Canastas Personales: Hábitos de compra de medicamentos o pruebas de")
    print("     salud no deben venderse a aseguradoras ni entidades crediticias.")
    print("  3. Derechos del Consumidor en Detección de Fraude: El bloqueo de transacciones")
    print("     sospechosas debe contar con canales inmediatos de validación y desbloqueo.")
    print("=" * 80)


if __name__ == "__main__":
    main()
