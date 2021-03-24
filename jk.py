from tkinter import *
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

class window:
    def __init__(self, master):
        self.master = master

        self.AFFMvalue = StringVar()
        self.result = StringVar()
        self.res=StringVar()


        canvas2 = Canvas(self.master, width=150, height=150, bg="#17202a")

        canvas2.place(x=20, y=20)
        self.master.my_image = PhotoImage(file='logo.png')
        canvas2.create_image(0, 0, anchor="nw", image=self.master.my_image)

        self.ptilte = Label(self.master, text="             GDP PREDICTOR           ", bg="#DBE7FF",
                            font=('Imprint MT Shadow', 13, "bold"),
                            fg="#000055", relief="groove")
        self.ptilte.pack()
        Label(self.master, text="Department Of Computer Science & Engineering", font=('Imprint MT Shadow ', 12, "bold"),
              fg="#fdfefe", bg="#17202a").pack()
        AFFM = Label(self.master, text="GDP:", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        AFFMe1 = Entry(self.master, textvariable=self.AFFMvalue, fg="#000055", font=("", 11, "bold"), relief="sunken",
                       bd=5,
                       bg="#F4FCE3")
        AFFM.place(x=300, y=100)
        AFFMe1.place(x=400, y=100)






        lrgdp = Button(self.master, text="Get Values", command=self.calculategdplr, font=("ARIAL", 11, "bold"),
                       bg="#555555",
                       fg="white",
                       relief="raised", bd=5)
        lrgdp.place(x=350, y=250)

    def check(self):
        plist = [self.AFFMvalue.get()]
        flag = True
        for p in plist:
            if p == "" or p == " ":
                messagebox.showerror("Predicted", "You cannot leave entry blank", icon="error")
                flag = False
        return flag

    def calculategdplr(self):
        # importing data
        data = pd.read_csv('gdpgr.csv', sep=",")
        df = pd.DataFrame(data)

        # data cleaning process
        df.drop(columns='Year', inplace=True)
        df.fillna(df.mean(), inplace=True)
        x1 = df[['GDP']]
        y1 = df.drop('GDP', axis=1)

        # splitting data to 20% testing 80% training
        X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        res1=""

        flag =self.check()
        if flag:
            predicted_lr = model.predict([[float(self.AFFMvalue.get())]])
            for i in predicted_lr:
                self.result.set(i)

            messagebox.showinfo("Predicted", "Values : "+self.result.get(), icon="info")
            self.clear()


    def clear(self):
        self.AFFMvalue.set('')



mw = Tk()
mw.geometry("800x500")
mw.configure(background="#17202a", border=0.5)
mw.title("COMPUTER SCIENCE DEPARTMENT")
myapp = window(mw)
mw.mainloop()
