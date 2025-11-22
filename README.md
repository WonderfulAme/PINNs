# 📌 Physics-Informed Neural Networks (PINNs) – Solución de EDPs clásicas

Este repositorio contiene una implementación sencilla y extensible de **Physics-Informed Neural Networks (PINNs)** en PyTorch, diseñada para resolver varias ecuaciones diferenciales parciales (EDPs) sin necesidad de datos etiquetados.

El objetivo del repositorio es servir como una **plantilla práctica** para:
- Resolver EDPs con PINNs
- Probar nuevas ecuaciones

## 📁 Estructura del repositorio

```
PINNs/
│
├── PINN.py                # Implementación principal del modelo PINN
├── utilities.py           # Funciones auxiliares para visualización
├── main_eq.ipynb          # Ejecutor general para múltiples PDEs
├── diffusion_eq.ipynb     # Ejemplo enfocado en la ecuación de difusión
├── proy-final.pdf         # Informe final explicando como funciona la PINN y resultados
│
├── README.md
```

## 📘 Explicación por archivo

## **1. `PINN.py` (archivo principal del proyecto)**
Este archivo implementa la clase *PINNSolver*. Responsable de:
- Construir la red neuronal (MLP con activación Tanh)
- Generar datos de entrenamiento:
  - condiciones de frontera
  - condiciones iniciales
  - *collocation points* mediante Latin Hypercube Sampling  
- Calcular derivadas automáticas con `torch.autograd`
- Definir la función de pérdida física:
  - residuo de la PDE  
  - error en las condiciones de frontera  
- Entrenar la red con Adam  
- Evaluar y graficar los resultados y errores
  
---

## **2. `utilities.py`**
Incluye funciones para visualizar soluciones en 2D y 3D:

- `plot3D`
- `plot3D_Matrix`
  
---

## **3. `main_eq.ipynb`**
El *motor general* del repositorio.  
Permite ejecutar cualquiera de las EDPs disponibles:

```
"heat", "diffusion", "source", "wave", "burgers", "laplace2D"
```

Cada ecuación (de estos casos) cuenta con:
- solución exacta
- función de residuo de la PDE
- configuración del dominio
- número de pasos de entrenamiento

Para usarlo, basta con modificar:

```python
CASE = "diffusion"
````

---
## **4. `diffusion_eq.ipynb`**

Notebook simple y limpio para estudiar una sola ecuación: la **ecuación de difusión**. Perfecto para entender el funcionamiento del PINN sin distracciones.

---

# ▶️ Cómo usar este repositorio

## **1. Instalar dependencias**

```bash
pip install

torch
numpy
matplotlib
pyDOE
jupyter
```

---

## **2. Ejecutar un caso**

Abrir:

```
main_eq.ipynb
```

Seleccionar el PDE deseado:

```python
CASE = "heat"
```

Ejecutar todas las celdas.

La PINN:

* construirá los puntos de entrenamiento
* entrenará por N iteraciones
* graficará:
  
  * solución exacta
  * solución predicha
  * error absoluto

Además, se imprimen métricas cuantitativas:

* Error absoluto máximo
* Error medio
* Error relativo $L_2$

---

# 🛠️ Agregar una nueva EDP

Para añadir una ecuación personalizada:

1. Ir a `main_eq.ipynb`
2. Crear una función con:

```python
def case_miecuacion():
    def exact(x,t):
        return ...

    def residual(x_col, u_t, u_xx):
        return ...

    settings = dict(
        x_min=...,
        x_max=...,
        t_min=...,
        t_max=...,
        steps=20000
    )

    return exact, residual, settings
```

3. Agregarla al selector:

```python
elif CASE == "miecuacion":
    exact_solution, pde_equation, config = case_miecuacion()
```

¡Y listo!
---

# 🎯 ¿Qué hace este código?

Cuando ejecutas un caso, el sistema:

1. Define la ecuación diferencial y su solución exacta
2. Construye un MLP totalmente conectado
3. Genera los puntos de entrenamiento
4. Calcula derivadas como: $u_x,\ u_t,\ u_{xx},\ u_{tt}$
5. Construye la pérdida: $\mathcal{L} = \mathcal{L}*{BC} + \mathcal{L}*{PDE}$
6. Entrena la red
7. Evalúa la solución
8. Genera gráficas y métricas
---

## Recursos / Bibliografía

- [Physics-Informed Neural Networks (YouTube)](https://www.youtube.com/watch?v=-zrY7P2dVC4)
- [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [PINNs GitHub Repository](https://github.com/jayroxis/PINNs)
- [Physics-Informed Neural Networks: A Simple Tutorial with PyTorch (Medium)](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a)
- [Solving Differential Equations with Neural Networks (Medium)](https://medium.com/data-science/solving-differential-equations-with-neural-networks-afdcf7b8bcc4)
- [What Is a Physics-Informed Neural Network? (Ben Moseley Blog)](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
- [Learning Physics Informed Machine Learning](https://www.youtube.com/watch?v=AXXnSzmpyoI)
