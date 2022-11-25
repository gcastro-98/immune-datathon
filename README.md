# immune-datathon
Immune 2022 datathon edition: implementation of the solution on the churn prediction challenge


## Data

Tenemos disponibles 3 datasets, los cuales están descritos en el apartado data de la competición:

- ``train.csv``: conjunto de entrenamiento, que se usará para entrenar modelos de predicción
- ``test.csv``: conjunto de test, con distintos clientes, para los que se deberá predecir la etiqueta usando el modelo entrenado con el conjunto de entrenamiento
- ``sample_submission.csv``: ejemplo de fichero de entrega. 

Los datos están cargados en ficheros CSV y contienen los siguientes campos:

- **customer_id** - identificador de cliente. Cada fila tiene un id único, y corresponde a un cliente diferente.
- **customer_age** - edad del cliente
- **education_level** - nivel de formación del cliente. Posibles valores:
	- graduate: con grado universitario
	- high school: instituto
	- uneducated: sin formación
	- college: en la universidad
	- post-graduate: con máster universitario
	- phd: doctorado
- **marital_status**: estado civil. Posibles valores:
	- married: casado
	- single: soltero
	- divorced: divorciado
- **income_category**: ingresos del cliente, por categoría. Posibles valores:
	- <30k€: menos de 30.000 euros anuales
	- 30k€-50k€: entre 30 y 50 mil euros anuales
	- 50k€-70k€ : entre 50 y 70 mill euros anuales
	- 70k€-110k€: entre 70 y 110 mil euros anuales
	- +110k€: más de 110 mil euros anuales
- **number_products_customer**: número de productos que el cliente tiene contratado con el banco.
- **weeks_tenure**: número de semanas desde que el cliente dio de alta su primer producto con el banco.
- **contacts_last_12mths**: número de interacciones entre el cliente y el banco en el último año. Una interacción puede iniciarse por ambos lados. Por ejemplo, un cliente puede llamar a su gestor para pedir información sobre más productos, o el banco puede contactar al cliente con una campaña comercial.
- **credit_limit**: límite de crédito en la tarjeta.
- **card_class**: tipo de tarjeta. Cada tipo tiene una serie de ventajas. Posibles valores:
	- red
	- red plus
	- red unlimited
	- premium
- **inactive_months_last_12mths**: número de meses en el último año en los que el cliente no ha utilizado la tarjeta de crédito.
- **total_revolving_balance**: cantidad de deuda que no se ha pagado en un ciclo, y que pasa a la cuenta del siguiente, con una consecuente subida de interés.
- **count_transactions**: número de transacciones realizadas con la tarjeta en el último año.
- **transactions_amount**: cantidad de dinero involucrado en las transacciones del último año.
- **change_transaction_amt_last_3mths**: diferencia relativa de cantidad de dinero gastado en los 3 últimos meses. Un valor mayor que 1 indica un aumento en el gasto.
- **change_transaction_count_last_3mths**: diference relativa de transacciones en los últimos 3 meses. Un valor mayor que 1 indica un aumento en el número de transacciones.

- **churn**: etiqueta, _label_. Tendrá valor 0 si el cliente sigue con la entidad, y un 1 si ha dado de baja su tarjeta de crédito. Este campo solo está presente en el conjunto train.csv.