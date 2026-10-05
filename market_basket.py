"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 24: Minería de Datos - Detective de Supermercado (FP-Growth, K-Means & Isolation Forest)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import sys
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
except ImportError:
    TransactionEncoder = None


def generar_transacciones_supermercado(n_transacciones: int = 5000, random_seed: int = 42) -> List[List[str]]:
    """
    Genera 5,000 canastas de mercado con patrones de compra reales:
    - Desayuno: pan, leche, cafe, mantequilla, mermelada
    - Carne asada: carne, carbon, cerveza, salsa, tortillas
    - Limpieza: detergente, suavizante, cloro, papel_higienico
    - Compras cotidianas variadas
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    patrones = [
        # Patrón Desayuno (Alta probabilidad de asociación pan -> leche, cafe -> azucar)
        {"items": ["pan_blanco", "leche_entera", "cafe_soluble", "mantequilla", "mermelada"], "prob": 0.35},
        # Patrón Reunión / Asado
        {"items": ["carne_res", "carbon_vegetal", "cerveza_clara", "salsa_picante", "tortillas_maiz"], "prob": 0.25},
        # Patrón Limpieza
        {"items": ["detergente_polvo", "suavizante", "cloro_desinfectante", "papel_higienico", "jabon_barra"], "prob": 0.20},
        # Patrón Alacena básica
        {"items": ["arroz_grano", "frijol_negro", "aceite_vegetal", "pasta_sopa", "pure_tomate"], "prob": 0.30},
        # Patrón Saludable
        {"items": ["manzana_roja", "platano", "avena_integral", "yogurt_natural", "miel_abeja"], "prob": 0.18}
    ]

    items_sueltos = ["galletas", "refresco_cola", "papas_fritas", "chocolates", "agua_mineral", "queso_fresco"]

    transacciones = []
    for _ in range(n_transacciones):
        canasta = set()
        for p in patrones:
            if random.random() < p["prob"]:
                # Tomar entre 2 y todos los artículos del combo
                k = random.randint(2, len(p["items"]))
                canasta.update(random.sample(p["items"], k))
        
        # Añadir de 0 a 2 ítems aleatorios
        num_aleatorios = random.randint(0, 2)
        canasta.update(random.sample(items_sueltos, num_aleatorios))

        if len(canasta) == 0:
            canasta.add("pan_blanco")
            canasta.add("leche_entera")
            
        transacciones.append(sorted(list(canasta)))

    return transacciones


def minar_reglas_asociacion(transacciones: List[List[str]], min_support=0.08, min_confidence=0.55, min_lift=1.8):
    print("\n" + "=" * 75)
    print("  1. MINERÍA DE REGLAS DE ASOCIACIÓN (FP-GROWTH / APRIORI)")
    print("=" * 75)
    print(f"Filtros de minería: Soporte > {min_support} | Confianza > {min_confidence} | Lift > {min_lift}")

    if TransactionEncoder is not None:
        te = TransactionEncoder()
        te_ary = te.fit(transacciones).transform(transacciones)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

        # Usar FP-Growth por su eficiencia O(N) frente a Apriori O(2^k)
        itemsets_frecuentes = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
        print(f"[*] Conjuntos frecuentes encontrados (Soporte >= {min_support}): {len(itemsets_frecuentes)}")

        if len(itemsets_frecuentes) == 0:
            print("[!] No se encontraron itemsets con el umbral especificado.")
            return pd.DataFrame()

        reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=min_confidence)
        
        # Filtrar por Lift estricto
        reglas_filtradas = reglas[reglas["lift"] > min_lift].sort_values(by="lift", ascending=False)
        print(f"[+] Reglas que cumplen Soporte > {min_support}, Confianza > {min_confidence} y Lift > {min_lift}: {len(reglas_filtradas)}\n")

        for idx, row in reglas_filtradas.head(10).iterrows():
            ant = list(row["antecedents"])
            con = list(row["consequents"])
            print(f"Regla: {ant} ---> {con}")
            print(f"  Soporte: {row['support']:.4f} | Confianza: {row['confidence']*100:.1f}% | Lift: {row['lift']:.3f}\n")

        return reglas_filtradas
    else:
        print("[*] 'mlxtend' no está instalado. Ejecutando motor Apriori nativo de alta precisión...")
        from collections import defaultdict
        N = len(transacciones)
        item_counts = defaultdict(int)
        for t in transacciones:
            for item in set(t):
                item_counts[item] += 1
        freq_1 = {item: count / N for item, count in item_counts.items() if count / N >= min_support}

        pair_counts = defaultdict(int)
        for t in transacciones:
            items_en_t = [item for item in set(t) if item in freq_1]
            for i in range(len(items_en_t)):
                for j in range(i + 1, len(items_en_t)):
                    pair = tuple(sorted([items_en_t[i], items_en_t[j]]))
                    pair_counts[pair] += 1

        reglas = []
        for (item_a, item_b), count in pair_counts.items():
            supp_ab = count / N
            if supp_ab >= min_support:
                conf_ab = supp_ab / freq_1[item_a]
                lift_ab = conf_ab / freq_1[item_b]
                if conf_ab >= min_confidence and lift_ab >= min_lift:
                    reglas.append(([item_a], [item_b], supp_ab, conf_ab, lift_ab))

                conf_ba = supp_ab / freq_1[item_b]
                lift_ba = conf_ba / freq_1[item_a]
                if conf_ba >= min_confidence and lift_ba >= min_lift:
                    reglas.append(([item_b], [item_a], supp_ab, conf_ba, lift_ba))

        reglas.sort(key=lambda x: x[4], reverse=True)
        print(f"[+] Reglas descubiertas que cumplen filtros: {len(reglas)}\n")
        for ant, con, supp, conf, lift in reglas[:10]:
            print(f"Regla: {ant} ---> {con}")
            print(f"  Soporte: {supp:.4f} | Confianza: {conf*100:.1f}% | Lift: {lift:.3f}\n")

        return reglas


def segmentar_clientes_kmeans(n_clientes: int = 1000):
    print("=" * 75)
    print("  2. SEGMENTACIÓN DE CLIENTES (CLUSTERING K-MEANS)")
    print("=" * 75)
    
    np.random.seed(42)
    # Generar 3 arquetipos de clientes
    # Grupo 1: Comprador Frecuente pero Ticket Bajo (ej. tienda de paso)
    c1_frec = np.random.normal(20, 3, 400).clip(10, 35)
    c1_ticket = np.random.normal(120, 25, 400).clip(40, 250)

    # Grupo 2: Comprador Familiar Quincenal (Ticket Alto, Frecuencia Media)
    c2_frec = np.random.normal(4, 1.2, 400).clip(1, 8)
    c2_ticket = np.random.normal(1450, 200, 400).clip(800, 2200)

    # Grupo 3: Clientes Premium / Gourmet (Frecuencia Alta, Ticket Alto)
    c3_frec = np.random.normal(14, 2.5, 200).clip(8, 22)
    c3_ticket = np.random.normal(2600, 350, 200).clip(1800, 4000)

    frecuencias = np.concatenate([c1_frec, c2_frec, c3_frec])
    tickets = np.concatenate([c1_ticket, c2_ticket, c3_ticket])

    df_clientes = pd.DataFrame({
        "visitas_mensuales": frecuencias.round(1),
        "ticket_promedio_mxn": tickets.round(2)
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clientes)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clientes["cluster"] = kmeans.fit_predict(X_scaled)

    perfiles = df_clientes.groupby("cluster").mean()
    print("Perfiles de Clientes Identificados:")
    nombres_cluster = {
        0: "Comprador de Conveniencia Diaria",
        1: "Despensa Familiar Quincenal",
        2: "Cliente VIP / Alta Rentabilidad"
    }
    for c_id, row in perfiles.iterrows():
        n_miembros = (df_clientes["cluster"] == c_id).sum()
        print(f"  Cluster {c_id} ({nombres_cluster.get(c_id, 'Segmento')}) [{n_miembros} clientes]:")
        print(f"    * Visitas promedio: {row['visitas_mensuales']:.1f} al mes")
        print(f"    * Ticket promedio: ${row['ticket_promedio_mxn']:.2f} MXN\n")


def detectar_transacciones_anomalas(transacciones: List[List[str]]):
    print("=" * 75)
    print("  3. DETECCIÓN DE TRANSACCIONES ANÓMALAS (ISOLATION FOREST)")
    print("=" * 75)

    # Vector de características por transacción: [longitud_canasta, valor_estimado, diversidad_items]
    np.random.seed(42)
    X = []
    for t in transacciones:
        cant_items = len(t)
        # Precio simulado promedio de $45 por producto con varianza
        valor_total = sum(np.random.uniform(20, 80) for _ in t)
        X.append([cant_items, valor_total])

    # Insertar anomalías deliberadas (compras masivas de acaparamiento o tickets corruptos)
    X.append([65, 9500.0]) # Acaparamiento sospechoso
    X.append([1, 15000.0])  # Error en sistema (1 producto a $15,000)
    X.append([70, 11200.0])

    X_mat = np.array(X)
    
    # Contaminación esperada: 0.5% de transacciones anómalas
    iso_forest = IsolationForest(contamination=0.005, random_state=42)
    predicciones = iso_forest.fit_predict(X_mat) # -1 anómalo, 1 normal

    indices_anomalos = np.where(predicciones == -1)[0]
    print(f"[*] Total de transacciones auditadas: {len(X)}")
    print(f"[!] Transacciones marcadas como anómalas / sospechosas: {len(indices_anomalos)}\n")

    print("Muestra de casos anómalos detectados para auditoría:")
    for idx in indices_anomalos[:5]:
        print(f"  * Transacción #{idx}: {X_mat[idx, 0]:.0f} productos, Importe: ${X_mat[idx, 1]:,.2f} MXN (Alerta de fraude/acaparamiento)")

    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS EN MINERÍA DE DATOS Y PERFILADO DE CONSUMO")
    print("=" * 75)
    print("""
    1. Precios Dinámicos Discriminatorios: La segmentación de clientes no debe ser
       empleada para fijar precios abusivos o privar de productos básicos a ciertos sectores.
    2. Privacidad de Canastas de Compra: Las compras pueden revelar inadvertidamente
       condiciones médicas (embarazo, enfermedades crónicas) o hábitos religiosos.
    3. Falsos Positivos en Fraude: Bloquear tarjetas o transacciones sin un mecanismo
       ágil de apelación perjudica la confianza y los derechos del consumidor.
    """)


def main():
    print("=" * 75)
    print("  TECNM / ITSU - DETECTIVE DE DATOS DE SUPERMERCADO")
    print("=" * 75)

    print("[*] Generando 5,000 transacciones de supermercado con hábitos realistas...")
    transacciones = generar_transacciones_supermercado(5000)
    print(f"[+] Dataset sintético listo ({len(transacciones)} registros).")

    minar_reglas_asociacion(transacciones, min_support=0.08, min_confidence=0.55, min_lift=1.8)
    segmentar_clientes_kmeans()
    detectar_transacciones_anomalas(transacciones)


if __name__ == "__main__":
    main()
