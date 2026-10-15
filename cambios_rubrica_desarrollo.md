# Cambios para cumplir la rúbrica de Desarrollo (objetivo: 100/100)

Revisé la rúbrica de **Desarrollo** y el notebook `reto.ipynb`. La calificación se divide en dos criterios:

- **B1. Ciclo funcional y modificación justificada — 70 puntos**
- **B2. Reproducción y comparación controlada — 30 puntos**

El punto clave es que comparar únicamente las probabilidades de mutación `0.25` y `0.50` no satisface la modificación de operador que exige la entrega de Desarrollo.

Debes separar los experimentos:

- **Desarrollo:** algoritmo base frente a variante, ambos con mutación `0.25`.
- **Entrega final:** una misma versión del algoritmo con mutación `0.25` frente a `0.50`.

La rúbrica indica que no es necesario que la variante obtenga un resultado mejor. Puede alcanzar la máxima calificación si el cambio, la comparación y la interpretación están correctamente justificados y documentados.

## 1. Modificación recomendada: selección por torneo de tres

Recomiendo cambiar el operador de selección de torneo de dos individuos a torneo de tres. Es un cambio sustantivo que la rúbrica contempla como ejemplo.

### Justificación

- **Base:** se eligen dos individuos al azar y gana el de menor aptitud `F`.
- **Variante:** se eligen tres individuos distintos y gana el de menor aptitud `F`.

**Hipótesis:** el torneo de tres podría aumentar la presión de selección y favorecer la propagación de candidatos con menor aptitud. Esto podría ayudar a encontrar asignaciones factibles con menor consumo de agua potable. Sin embargo, también podría reducir la diversidad y hacer que la búsqueda se estanque antes.

No se debe afirmar que la variante es mejor antes de ejecutar y analizar los resultados.

## 2. Conserva el algoritmo base

No borres ni reemplaces las funciones originales:

- `tournament`
- `crossover`
- `mutate`
- `next_generation`
- `run_ga`

La base debe seguir disponible y ser identificable. La variante se añade por separado para poder ejecutar ambas versiones.

## 3. Agrega una función de selección por torneo de tres

Agrega una nueva celda después de la función de selección original. El código siguiente supone que el notebook ya define `evaluate` y que cada individuo puede copiarse con `.copy()`.

```python
def tournament_variant(population, rng):
    """Selecciona al mejor de tres candidatos distintos."""
    if len(population) < 3:
        raise ValueError("Se necesitan al menos tres individuos.")

    indices = rng.choice(len(population), size=3, replace=False)
    winner = min(
        (population[i] for i in indices),
        key=lambda individual: evaluate(individual)["fitness"]
    )
    return winner.copy()
```

La función no modifica la población original y mantiene el criterio de que una aptitud menor es mejor.

## 4. Agrega una función de siguiente generación para la variante

Agrega otra celda después de la función `next_generation` original. La variante debe usar la nueva selección, pero mantener la cruza, mutación, reemplazo generacional y elitismo de la base.

```python
def next_generation_variant(population, probability, rng):
    """Genera la siguiente población usando torneo de tres."""
    elite = min(
        population,
        key=lambda individual: evaluate(individual)["fitness"]
    ).copy()

    replacement = [elite]

    while len(replacement) < len(population):
        parent_a = tournament_variant(population, rng)
        parent_b = tournament_variant(population, rng)

        child = crossover(parent_a, parent_b, rng)
        child = mutate(child, probability, rng)

        replacement.append(child)

    assert len(replacement) == len(population)
    assert all(
        np.all(individual >= 0) and np.all(individual <= LIMITS)
        for individual in replacement
    )

    # La élite debe conservarse sin alteraciones.
    np.testing.assert_array_equal(replacement[0], elite)

    return replacement
```

Esta función conserva una población del mismo tamaño y protege al mejor individuo. Verifica que `LIMITS`, `np`, `evaluate`, `crossover` y `mutate` estén definidos previamente en el notebook.

## 5. Agrega una ejecución independiente para la variante

La función `run_ga` original utiliza `next_generation`; por eso, definir solamente la nueva selección no cambia el ciclo completo. Añade esta función para ejecutar la variante de principio a fin.

```python
def run_ga_variant(
    population_size=20,
    generations=20,
    mutation_probability=0.25,
    seed=42
):
    """Ejecuta el algoritmo con selección por torneo de tres."""
    if not isinstance(population_size, int) or population_size < 3:
        raise ValueError("La población debe tener al menos tres individuos.")
    if not isinstance(generations, int) or generations < 1:
        raise ValueError("Las generaciones deben ser positivas.")
    if not 0 <= mutation_probability <= 1:
        raise ValueError("La probabilidad debe estar entre 0 y 1.")

    rng = np.random.default_rng(seed)

    population = [
        random_individual(rng)
        for _ in range(population_size)
    ]

    history = [
        min(evaluate(individual)["fitness"] for individual in population)
    ]

    for _ in range(generations):
        population = next_generation_variant(
            population, mutation_probability, rng
        )
        history.append(
            min(evaluate(individual)["fitness"] for individual in population)
        )

    best = min(
        population,
        key=lambda individual: evaluate(individual)["fitness"]
    ).copy()

    return {
        "genes": best,
        "metrics": evaluate(best),
        "history": np.array(history)
    }
```

La inicialización usa la misma semilla y la misma regla de generación que la base. Después de la inicialización, las secuencias aleatorias pueden diferir porque los torneos consumen números aleatorios de manera distinta. Eso es esperable.

**Importante:** comprueba que las claves del diccionario devuelto por `evaluate` sean exactamente `fitness`, `treated`, `potable`, `excess` y `feasible`. Si el notebook usa nombres distintos, adapta las referencias del código a los nombres reales.

## 6. Ejecuta la comparación de Desarrollo

Para Desarrollo, compara la base con la variante usando las mismas condiciones:

- Población: 20 individuos.
- Generaciones: 20.
- Semilla: 42.
- Probabilidad de mutación: `0.25`.
- Mismos datos, dominios, aptitud y capacidad.
- Misma regla de inicialización.
- Mismo reemplazo generacional y elitismo.

Agrega una celda como la siguiente después de definir las funciones nuevas:

```python
# Experimento de Desarrollo: base vs. variante
CONFIG_DESARROLLO = {
    "population_size": 20,
    "generations": 20,
    "mutation_probability": 0.25,
    "seed": 42
}

# Versión base: torneo de dos
base_desarrollo = run_ga(**CONFIG_DESARROLLO)

# Variante: torneo de tres
variante_desarrollo = run_ga_variant(**CONFIG_DESARROLLO)

for nombre, resultado in [
    ("Base: torneo de 2", base_desarrollo),
    ("Variante: torneo de 3", variante_desarrollo)
]:
    m = resultado["metrics"]

    print(f"\\n{nombre}")
    print("Cromosoma:", resultado["genes"].tolist())
    print("Fracciones:", (resultado["genes"] / 10).tolist())
    print(f"Agua tratada T: {m['treated']:.3f} m³")
    print(f"Agua potable P: {m['potable']:.3f} m³")
    print(f"Aptitud F: {m['fitness']:.8f}")
    print(f"Exceso E: {m['excess']:.8f} m³")
    print("Factible:", m["feasible"])
```

Este ejemplo presupone que `run_ga` acepta esos nombres de argumentos y devuelve un diccionario con las claves indicadas. Confirma la firma y estructura de salida en tu notebook antes de ejecutarlo.

## 7. Construye una tabla comparativa

La tabla debe presentar los resultados reales de ambas ejecuciones, no valores inventados.

```python
import pandas as pd

def fila_comparacion(nombre, resultado):
    m = resultado["metrics"]
    genes = resultado["genes"]

    return {
        "Version": nombre,
        "Cromosoma": tuple(int(g) for g in genes),
        "Agua tratada T (m3)": m["treated"],
        "Agua potable P (m3)": m["potable"],
        "Aptitud F": m["fitness"],
        "Exceso E (m3)": m["excess"],
        "Genes validos": bool(
            np.all(genes >= 0) and np.all(genes <= LIMITS)
        ),
        "Factible": m["feasible"]
    }

tabla_desarrollo = pd.DataFrame([
    fila_comparacion("Base: torneo de 2", base_desarrollo),
    fila_comparacion("Variante: torneo de 3", variante_desarrollo)
])

display(tabla_desarrollo.round(6))
```

La comparación de consumo de agua potable debe considerar únicamente candidatos factibles. Un candidato con menor `P` pero con exceso de capacidad no representa una mejora factible.

## 8. Documenta la hipótesis en una celda Markdown

Puedes usar el siguiente borrador y ajustar la interpretación después de observar los resultados:

### Experimento de Desarrollo: modificación de la selección

**Limitación identificada.** La versión base selecciona a los padres mediante un torneo de dos individuos. Aunque este mecanismo favorece a los candidatos con menor aptitud, la presión de selección puede ser limitada en algunas poblaciones.

**Modificación propuesta.** En la variante se aumenta el tamaño del torneo de dos a tres candidatos. Se seleccionan tres individuos distintos y se elige al de menor aptitud. Se modifica únicamente el operador de selección; la cruza de un punto, la mutación por gen, el reemplazo generacional y el elitismo conservan su implementación original.

**Hipótesis.** Esperamos que el torneo de tres aumente la presión de selección y favorezca la propagación de candidatos con menor aptitud. Esto podría ayudar a encontrar asignaciones factibles que utilicen menos agua potable. Sin embargo, una presión de selección mayor también puede reducir la diversidad de la población y limitar la exploración.

**Condiciones de comparación.** Ambas versiones utilizan los mismos datos de marzo de 2025, los mismos dominios de los genes, una capacidad didáctica de 1,410 m³, una población de 20 individuos, 20 generaciones, una probabilidad de mutación de 0.25 y la semilla 42. Se conserva la misma regla de inicialización. La única modificación intencional es el tamaño del torneo de selección.

**Interpretación de resultados.** Compararemos los cromosomas finales, el agua tratada, el agua potable, la aptitud, el exceso de capacidad y la factibilidad. Una menor aptitud no se interpretará automáticamente como una mejora física: también comprobaremos que los genes sean válidos y que el consumo de agua tratada no exceda la capacidad.

**Limitaciones.** Una comparación con una sola semilla describe el resultado de una ejecución concreta. No demuestra que el torneo de tres sea superior en general. Los resultados pueden apoyar, contradecir o no permitir distinguir la hipótesis. Se conservará la conclusión que corresponda a la evidencia observada.

## 9. Comprueba la reproducibilidad, los dominios y el elitismo

Tu notebook ya incluye comprobaciones para la base. Amplíalas para la variante:

```python
# Reproducción independiente de ambas versiones
base_repetida = run_ga(**CONFIG_DESARROLLO)
variante_repetida = run_ga_variant(**CONFIG_DESARROLLO)

np.testing.assert_array_equal(
    base_desarrollo["genes"], base_repetida["genes"]
)
np.testing.assert_array_equal(
    base_desarrollo["history"], base_repetida["history"]
)

np.testing.assert_array_equal(
    variante_desarrollo["genes"], variante_repetida["genes"]
)
np.testing.assert_array_equal(
    variante_desarrollo["history"], variante_repetida["history"]
)

# La mejor aptitud no debe empeorar si se protege la élite.
assert np.all(np.diff(base_desarrollo["history"]) <= 1e-12)
assert np.all(np.diff(variante_desarrollo["history"]) <= 1e-12)

# Los resultados finales deben respetar los dominios.
for resultado in [base_desarrollo, variante_desarrollo]:
    genes = resultado["genes"]
    assert np.all(genes >= 0)
    assert np.all(genes <= LIMITS)
    assert len(resultado["history"]) == 21

print("Reproducción, dominios y elitismo comprobados.")
```

Esta prueba presupone que el historial almacena la mejor aptitud de la población inicial y de cada una de las 20 generaciones. Comprueba que la longitud sea 21 y que la mejor aptitud no empeore con el elitismo. No demuestra que el algoritmo haya encontrado el óptimo global.

## 10. Agrega una gráfica para Desarrollo

Conserva la gráfica que compara `p=0.25` y `p=0.50` para la entrega final. Para Desarrollo, crea una gráfica separada:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    base_desarrollo["history"],
    marker=".",
    label="Base: torneo de 2"
)
ax.plot(
    variante_desarrollo["history"],
    marker=".",
    label="Variante: torneo de 3"
)

ax.set_xlabel("Generación")
ax.set_ylabel("Mejor aptitud F (menor es mejor)")
ax.set_title("Desarrollo: selección por torneo de 2 vs. 3")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()
```

Debajo de la gráfica, responde con base en los resultados reales:

1. ¿Qué versión obtuvo menor aptitud final?
2. ¿Ambos candidatos son factibles?
3. ¿Cuál utiliza menos agua potable entre los candidatos factibles?
4. ¿En qué generación aparece una mejora o una meseta?
5. ¿Los resultados apoyan la hipótesis o no permiten distinguir las versiones?
6. ¿Por qué una sola semilla no basta para demostrar superioridad general?

No necesitas que las curvas sean distintas ni que la variante gane para obtener la puntuación máxima.

## 11. Lista de verificación para la entrega

### B1 — Ciclo funcional y modificación justificada (70 puntos)

- [ ] Conservar identificable el algoritmo base original.
- [ ] Implementar un cambio sustantivo en un solo operador.
- [ ] Explicar la limitación, la hipótesis y el mecanismo esperado.
- [ ] Ejecutar el ciclo completo de la variante.
- [ ] Verificar población constante, dominios válidos y élite protegida.
- [ ] Explicar la función de cada paso relevante del ciclo genético.

### B2 — Reproducción y comparación controlada (30 puntos)

- [ ] Ejecutar ambas versiones desde cero con semilla 42.
- [ ] Mantener las mismas condiciones experimentales y la misma inicialización.
- [ ] Presentar ambos cromosomas finales.
- [ ] Comparar `T`, `P`, `F`, `E`, dominios y factibilidad.
- [ ] Repetir ambas ejecuciones y comprobar la reproducibilidad.
- [ ] Interpretar la evidencia y explicar el límite de usar una sola semilla.

### Archivos y entrega

- [ ] Reiniciar el kernel y ejecutar todas las celdas en orden.
- [ ] Conservar las salidas reales y corregir errores.
- [ ] Entregar la libreta acumulativa `.ipynb`.
- [ ] Entregar `pyproject.toml` y `uv.lock`.
- [ ] Integrar las correcciones pertinentes de Investigación.
- [ ] Conservar el registro breve de rotación del equipo.

## 12. Diferencia entre Desarrollo y entrega final

| Elemento | Desarrollo | Entrega final |
|---|---|---|
| Comparación | Base vs. variante | Dos configuraciones de una misma versión |
| Operador | Torneo de 2 vs. 3 | Se mantiene el operador elegido |
| Mutación | 0.25 en ambas versiones | 0.25 vs. 0.50 |
| Población | 20 | 20 |
| Generaciones | 20 | 20 |
| Semilla | 42 | 42 |
| Evidencia | Tabla, hipótesis y reproducción | Explicación de una generación, gráfica e interpretación física |

La entrega final añade tres criterios: explicación del ciclo de una generación (40 puntos), comparación de probabilidades de mutación (35 puntos) e interpretación física de la asignación de agua (25 puntos). Conserva la gráfica actual para esa parte y explica las fracciones y volúmenes por edificio, la capacidad y los supuestos didácticos.

## Conclusión

No necesitas rehacer el notebook desde cero. El cambio central es añadir una variante real de selección, ejecutar la comparación base/variante con mutación `0.25`, y documentar los resultados de manera reproducible. Estos cambios responden a los descriptores de nivel IV de la rúbrica, aunque la calificación final dependerá de la correcta ejecución, de que el código coincida con el notebook real y de la evaluación docente.

**Antes de entregar:** ejecuta el notebook completo desde un kernel reiniciado y verifica que los nombres de funciones, argumentos y claves del evaluador coincidan con los de tu código real.
