from Code.UI import main as UI


if __name__ == "__main__":
    app = UI.QApplication(sys.argv)
    window = UI.MainWindow()
    window.show()
    sys.exit(app.exec())