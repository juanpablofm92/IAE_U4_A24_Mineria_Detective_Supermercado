# Actividad 24: Proyecto "Detective de Datos: Descubriendo Patrones Ocultos"

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética (Unidad 4)  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 🎯 Contexto y Misión

Como consultor en ciencia de datos para la cadena minorista **Supermercado ABC**, el objetivo es optimizar la disposición física de artículos en góndolas, diseñar promociones cruzadas (*cross-selling*) y detectar anomalías en transacciones de compra.

El proyecto aborda tres misiones clave:
1. **Análisis de Canasta de Compra (*Market Basket Analysis*):** Minería de reglas de asociación sobre 5,000 transacciones para descubrir qué productos se compran juntos frecuentemente, validando el descubrimiento de los patrones objetivo predefinidos:
   * `['Pan', 'Leche'] → ['Mantequilla']`
   * `['Cerveza', 'Papas'] → ['Dulces']`
   * `['Carne', 'Verduras'] → ['Salsa']`
2. **Segmentación de Clientes (*Customer Clustering*):** Agrupamiento conductual mediante **K-Means** identificando perfiles por frecuencia de visitas, ticket promedio ($ MXN) y tamaño de canasta.
3. **Detección de Anomalías (*Fraud & Outlier Detection*):** Detección no supervisada con **Isolation Forest** para aislar compras sospechosas por acaparamiento, montos desproporcionados o compras en horarios inusuales.

---

## 📋 Resultados de las Misiones

### Misión 1: Reglas de Asociación Descubiertas
* Umbrales de filtrado: Soporte $\ge 6.0\%$, Confianza $\ge 50.0\%$, Lift $\ge 1.60\times$.
* Reglas destacadas:
  * `['Detergente'] → ['Jabón']`: Soporte $10.72\%$, Confianza $81.71\%$, Lift **$6.079$**
  * `['Suavizante'] → ['Jabón']`: Soporte $10.44\%$, Confianza $81.31\%$, Lift **$6.050$**
  * `['Pan', 'Leche'] → ['Mantequilla']`: Soporte $8.40\%$, Confianza $74.20\%$, Lift **$2.650$**
  * `['Cerveza', 'Papas'] → ['Dulces']`: Soporte $7.15\%$, Confianza $68.40\%$, Lift **$2.420$**
  * `['Carne', 'Verduras'] → ['Salsa']`: Soporte $7.85\%$, Confianza $71.10\%$, Lift **$2.510$**

### Misión 2: Perfiles de Clientes Identificados (K-Means, $k=3$)
* **Cluster 0 (Comprador de Conveniencia Diaria) [400 clientes]:** Visitas altas ($13.9$ días/mes), ticket bajo ($\$347.41$ MXN), pocos artículos ($3.2$ productos). Estrategia: promociones exprés en cajas.
* **Cluster 1 (Despensa Familiar Quincenal) [300 clientes]:** Visitas medias ($3.9$ días/mes), ticket balanceado ($\$2,093.08$ MXN), volumen medio ($9.5$ productos). Estrategia: cupones por volumen en abarrotes.
* **Cluster 2 (Cliente VIP / Alto Ticket) [300 clientes]:** Visitas bajas ($2.0$ días/mes), gasto masivo ($\$3,802.63$ MXN), canasta grande ($14.0$ productos). Estrategia: membresías premium y envíos a domicilio.

### Misión 3: Detección de Anomalías (Isolation Forest)
* Se auditaron $5,000$ transacciones, identificando **$50$ casos sospechosos** ($1.0\%$ de contaminación).
* Detección de patrones de fraude o acaparamiento mayorista:
  * Transacción #2411: 27 artículos, $\$5,660.59$ MXN (volumen atípico).
  * Transacción #892: 33 artículos, $\$4,987.04$ MXN (hora inusual $02:18$ am).

---

## 📂 Estructura del Repositorio

```text
IAE_U4_A24_Mineria_Detective_Supermercado/
├── market_basket.py           # Pipeline integral con la plantilla requerida y las 3 misiones
├── clusters_supermercado.png  # Gráfica de dispersión de la segmentación K-Means
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación técnica completa
```

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar análisis de canasta, segmentación y detección de anomalías
python market_basket.py
```

---

## ⚖️ Consideraciones Éticas en Analítica de Retail

1. **Rechazo a la Discriminación de Precios (*Dynamic Price Gouging*):** Los modelos de segmentación no deben emplearse para alterar artificialmente los precios de alimentos de primera necesidad en sucursales de zonas vulnerables.
2. **Confidencialidad de la Canasta Básica:** Los hábitos de consumo pueden inferir condiciones médicas (embarazo, medicación) o religiosas, por lo que los identificadores de pago deben anonimizarse.
3. **Derechos del Consumidor en Prevención de Fraude:** Los bloqueos preventivos por Isolation Forest deben contar con mecanismos ágiles de aclaración presencial y digital sin estigmatizar al comprador.
