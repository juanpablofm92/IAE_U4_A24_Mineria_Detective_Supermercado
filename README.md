# Actividad 24: Detective de Datos de Supermercado (FP-Growth, K-Means & Isolation Forest)

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto aborda la analítica integral de transacciones de venta en un entorno de autoservicio (*Market Basket Analysis*) implementando tres técnicas fundamentales de minería de datos y aprendizaje no supervisado:

1. **Minería de Reglas de Asociación (FP-Growth):**
   - Extracción de dependencias del tipo $\{\text{Antecedente}\} \rightarrow \{\text{Consecuente}\}$.
   - Filtrado por umbrales mínimos: Soporte ($>0.08$), Confianza ($>0.55$) y Lift ($>1.8$).
2. **Segmentación de Comportamiento de Clientes (K-Means):**
   - Agrupamiento de compradores según visitas mensuales y ticket promedio.
   - Identificación de 3 arquetipos de consumidores (Conveniencia diaria, Despensa quincenal y Clientes VIP).
3. **Detección de Transacciones Anómalas (Isolation Forest):**
   - Aislamiento de tickets atípicos en espacio bidimensional (volumen de artículos e importe económico) para detectar fraude o acaparamiento.

---

## 🚀 2. Instalación y Ejecución

```bash
pip install -r requirements.txt
python market_basket.py
```

---

## 📈 3. Métricas de Reglas de Asociación

* **Soporte:** $S(A \rightarrow B) = P(A \cup B)$
* **Confianza:** $C(A \rightarrow B) = P(B \mid A) = \frac{P(A \cup B)}{P(A)}$
* **Lift:** $\text{Lift}(A \rightarrow B) = \frac{P(A \cup B)}{P(A) \cdot P(B)}$ (indica el grado de correlación positiva sobre la independencia estadística).

---

## ⚖️ 4. Consideraciones Éticas en Minería de Datos

1. **Inferencia Invasiva de Privacidad:** El análisis de patrones de compra puede revelar inadvertidamente condiciones médicas, filiaciones ideológicas o hábitos privados de los consumidores.
2. **Discriminación Algorítmica de Precios:** Los modelos de segmentación no deben ser utilizados para imponer precios desfavorables a grupos vulnerables en artículos de primera necesidad.
