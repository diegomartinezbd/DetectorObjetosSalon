Sistema de Inventario Objetos en Salón

Para abrir el ejecutable puede ser desde acá: 

https://diegomartinezbd.github.io/DetectorObjetosSalon/

Este repositorio contiene la aplicación web y el modelo entrenado para realizar detección automática de objetos dentro de un salón de cómputo.

La aplicación permite subir una imagen del salón y obtiene:

Toda la lógica de inferencia corre localmente en el navegador mediante JavaScript, por lo que no requiere servidor.

📁 Estructura del repositorio
inventario/
│── index.html              # Aplicación web (interfaz principal)
│── script.js               # Lógica de detección y conteo
│── best.pt                 # Con el modelo modelo
│── README.md               # Este archivo


