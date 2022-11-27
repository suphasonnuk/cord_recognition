from model import Trainer


def validate():
    trainer = Trainer("models/binary_crossentropy.h5")
    trainer.train()
    trainer.save()
    trainer.save_architecture()

    results = trainer.validate()

# with open("training_results.csv", "w") as f:
#     f.write("loss_function, loss_value, categorical_accuracy, top_k_categorical_accuracy\n")


with open("validation_results.csv", "w") as f:
    f.write("instrument, categorical_accuracy\n")

# for loss_function in loss_functions[1:2]:
#     train(loss_function)
validate()
