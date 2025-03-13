from Code.UI import main as UI
from Code.UI import chat as chat
import sys

if __name__ == "__main__":
    app = chat.QApplication(sys.argv)
    window = chat.ChatWindow()
    window.show()
    sys.exit(app.exec())