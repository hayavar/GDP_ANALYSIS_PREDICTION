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
        self.MCEGWvalue = StringVar()
        self.THTCvalue = StringVar()
        self.FIREBvalue = StringVar()
        self.CSPSvalue = StringVar()
        self.result = StringVar()

        canvas2 = Canvas(self.master, width=150, height=150, bg="#17202a")

        canvas2.place(x=20, y=20)
        self.master.my_image = PhotoImage(file='logo.png')
        canvas2.create_image(0, 0, anchor="nw", image=self.master.my_image)

        self.ptilte = Label(self.master, text="             GDP PREDICTOR           ", bg="#DBE7FF",
                            font=('Imprint MT Shadow', 13, "bold"),
                            fg="#000055", relief="groove")
        self.ptilte.pack()
        Label(self.master, text="Department Of Computer Science & Engineering", font=('Imprint MT Shadow ', 16, "bold"),
              fg="#fdfefe", bg="#17202a").pack()

        AFFM = Label(self.master, text="AFFM:", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        AFFMe1 = Entry(self.master, textvariable=self.AFFMvalue, fg="#000055", font=("", 11, "bold"), relief="sunken",
                       bd=5,
                       bg="#F4FCE3")
        AFFM.place(x=300, y=100)
        AFFMe1.place(x=400, y=100)

        MCEGW = Label(self.master, text="MCEGW: ", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        MCEGWe2 = Entry(self.master, textvariable=self.MCEGWvalue, fg="#000055", font=("", 11, "bold"),
                        relief="sunken", bd=5, bg="#F4FCE3")

        MCEGW.place(x=275, y=160)
        MCEGWe2.place(x=400, y=160)

        THTC = Label(self.master, text="THTC: ", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        THTCe3 = Entry(self.master, textvariable=self.THTCvalue, fg="#000055", font=("", 11, "bold"),
                       relief="sunken", bd=5, bg="#F4FCE3")

        THTC.place(x=300, y=220)
        THTCe3.place(x=400, y=220)

        FIREB = Label(self.master, text="FIREB: ", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        FIREBe4 = Entry(self.master, textvariable=self.FIREBvalue, fg="#000055", font=("", 11, "bold"),
                        relief="sunken", bd=5, bg="#F4FCE3")

        FIREB.place(x=295, y=280)
        FIREBe4.place(x=400, y=280)

        CSPS = Label(self.master, text="CSPS: ", fg="#fdfefe", font=("", 14, "bold"), bg="#17202a")

        CSPSe5 = Entry(self.master, textvariable=self.CSPSvalue, fg="#000055", font=("", 11, "bold"),
                       relief="sunken", bd=5, bg="#F4FCE3")

        CSPS.place(x=300, y=340)
        CSPSe5.place(x=400, y=340)

        informationlabe = Label(self.master,
                                text="AFFM: Agriculture, forestry & fishing , miningand quarrying\n\nMCEGW: Manufacturing, construction, electricity, gas and water supply\n\n"
                                     "THTC: Trade, hotels, transport & communication\n\nFIREB: Financing, insurance,real estate and business services\n\n"
                                     "CSPS: Community social & personal services", fg="#fdfefe", font=("", 10, "bold"),
                                bg="#17202a")

        informationlabe.place(x=700, y=150)

        lrgdp = Button(self.master, text="Get GDP (LR)", command=self.calculategdplr, font=("ARIAL", 11, "bold"),
                       bg="#555555",
                       fg="white",
                       relief="raised", bd=5)
        lrgdp.place(x=350, y=420)

        rfrgdp = Button(self.master, text="Get GDP (RFR)", command=self.calculategdprfr, font=("ARIAL", 11, "bold"),
                        bg="#555555",
                        fg="white",
                        relief="raised", bd=5)
        rfrgdp.place(x=500, y=420)

        gdppredgraph = Button(self.master, text="GDP predicted graph", command=self.getgraph,
                              font=("ARIAL", 11, "bold"),
                              bg="#555555",
                              fg="white",
                              relief="raised", bd=5)
        gdppredgraph.place(x=300, y=500)

        gdpgraph = Button(self.master, text="GDP first 20 years graph", command=self.getyeargraph,
                          font=("ARIAL", 11, "bold"),
                          bg="#555555",
                          fg="white",
                          relief="raised", bd=5)
        gdpgraph.place(x=500, y=500)
    def check(self):
        plist = [self.AFFMvalue.get(), self.MCEGWvalue.get(),
                 self.THTCvalue.get(), self.FIREBvalue.get()
            , self.CSPSvalue.get()]
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
        y = df.GDP
        x = df.drop('GDP', axis=1)

        # splitting data to 20% testing 80% training
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        score=model.score(X_train,y_train)
        flag =self.check()
        if flag:
            predicted_lr = model.predict([[float(self.AFFMvalue.get()), float(self.MCEGWvalue.get()),
                                           float(self.THTCvalue.get()), float(self.FIREBvalue.get())
                                              , float(self.CSPSvalue.get())]])
            for i in predicted_lr:
                self.result.set(i)
            pred3 = model.predict(X_test)
            rmserfr = sqrt(mean_squared_error(y_test, pred3))
            messagebox.showinfo("Predicted", "GDP : {:.2f}".format(float(self.result.get()))+"\nAccuracy: "+str(score)+
                                "\nRMSE of LR : {:.2f}".format(rmserfr), icon="info")
            self.clear()

    def calculategdprfr(self):
        # importing data
        data = pd.read_csv('gdpgr.csv', sep=",")
        df = pd.DataFrame(data)

        # data cleaning process
        df.drop(columns='Year', inplace=True)
        df.fillna(df.mean(), inplace=True)
        y = df.GDP
        x = df.drop('GDP', axis=1)

        # splitting data to 20% testing 80% training
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Random Forest Regression
        regr = RandomForestRegressor(max_depth=7, n_estimators=130, oob_score=True, n_jobs=-1, random_state=0)
        regr.fit(X_train, y_train)
        score=regr.score(X_train,y_train)
        flag=self.check()
        if flag:
            predicted_rfr = regr.predict([[float(self.AFFMvalue.get()), float(self.MCEGWvalue.get()),
                                           float(self.THTCvalue.get()), float(self.FIREBvalue.get())
                                              , float(self.CSPSvalue.get())]])
            for i in predicted_rfr:
                self.result.set(i)
            pred3 = regr.predict(X_test)
            rmserfr = sqrt(mean_squared_error(y_test, pred3))
            messagebox.showinfo("Predicted", "GDP : {:.2f}".format(float(self.result.get()))+"\nAccuracy: "+str(score)+
                                "\nRMSE of RFR : {:.2f}".format(rmserfr), icon="info")
            self.clear()

    def clear(self):
        self.AFFMvalue.set('')
        self.MCEGWvalue.set('')
        self.THTCvalue.set('')
        self.FIREBvalue.set('')
        self.CSPSvalue.set('')

    def getgraph(self):
        import pandas as pd

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt

        # importing data
        data = pd.read_csv('gdpgr.csv', sep=",")
        df = pd.DataFrame(data)

        # data cleaning process
        df.drop(columns='Year', inplace=True)
        df.fillna(df.mean(), inplace=True)
        y = df.GDP
        x = df.drop('GDP', axis=1)

        # splitting data to 20% testing 80% training
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Random Forest Regression
        regr = RandomForestRegressor(max_depth=7, n_estimators=130, oob_score=True, n_jobs=-1, random_state=0)
        regr.fit(X_train, y_train)
        pred3 = regr.predict(X_test)

        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_lr = model.predict([[2, 2, 2, 2, 2]])
        predicted_lr_test = model.predict(X_test)

        # getting index value
        indexlist = []
        for p in y_test:
            indexlist.append(df[df['GDP'] == p].index.values[0])
        indexlist = [x + 1950 for x in indexlist]

        # Graph of Actual Value vs LR predicted value vs RFR predicted value
        p1 = plt.scatter(indexlist, predicted_lr_test, color="green")
        p2 = plt.scatter(indexlist, pred3, color="red")
        p3 = plt.scatter(indexlist, y_test, color="blue")

        plt.legend([p1, p2, p3], ['LR', 'RFR', 'ACTUAL'])
        plt.title("YEAR vs GDP ")
        plt.xlabel("YEAR")
        plt.ylabel("Prediction values")
        plt.show()
        self.clear()

    def getyeargraph(self):
        import pandas as pd
        import matplotlib.pyplot as plt

        data = pd.read_csv('gdpgr.csv', sep=",")
        df = pd.DataFrame(data)
        df = df.head(20)
        xaxis = df['Year']
        yaxis = [df['AFFM'], df['MCEGW'], df['THTC'], df['FIREB'], df['CSPS']]
        df = df.drop(columns=['Year', 'GDP'])
        labelnames = list(df.columns)

        p = [i for i in range(len(labelnames))]
        i = 0
        for y in yaxis:
            p[i] = plt.scatter(xaxis, y)
            i = i + 1

        plt.xticks(rotation=45)
        plt.legend(p, labelnames, bbox_to_anchor=(1.0, 1.0), loc='center')
        plt.show()


mw = Tk()
mw.geometry("1200x600")
mw.configure(background="#17202a", border=0.5)
mw.title("COMPUTER SCIENCE DEPARTMENT")
myapp = window(mw)
mw.mainloop()
