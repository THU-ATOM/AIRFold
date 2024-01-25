class ErrorMessage(str):
    def __repr__(self):
        return (
            "Error Message >>>>>>>>>\n"
            + str(self)
            + "\n<<<<<<<<<< Error Message"
        )
