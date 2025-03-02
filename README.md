# Image-classification-with-Federal-learning-in-the-dataset-CIARF10
In this project, we try to figure out how to construct and running a federal learning model in different devices.

From the name of federal learning, you can intuitively understand that this kind of method is in the way of learning from the cooperation of many devices.

We divide dataset into many small parts,create a server, and then the server hand out the data to different divices, which called clients,each client train a little part of data, after training, clients send reply to the server, the reply contains the parameters of the model just trined in the client.

By gathering replies from all clients, the server can eventually generate an aggregated model.

This kind of method can solve the question that even though we don't have a single device which has aboundant computing power, by dividing datasets and training seperately and simultaneously, we can significantly reduce the cost of time we need to pay for the model.

In this project, we will show 5 .py files:

data_loader.py: representes how to divide datasets into many small parts.

model.py: the file consists of the untrained model.

server.py: the file that constructs a server.

client.py: the file constructs clients.

local_training.py: the file provides a code that can train locally, where is a common way to train model, or we say 'non federal way'.

The main idea and more details will be expained in .py files.

To run the server, if you put our code in pycharm, you just need to click the server.py file and run it, or you can use the code below in your terminal:
`python server.py`

And to run a client, you need to use the code below in your terminal:
`python client.py --client_id 0`
And you can replace the client id number by order to activate more clients.

