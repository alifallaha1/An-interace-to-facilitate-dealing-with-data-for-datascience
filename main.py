import pandas as pd
from kivy.lang import Builder
from kivymd.uix.button import MDRaisedButton, MDFlatButton ,MDRectangleFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen , ScreenManager
from kivymd.uix.screen import MDScreen
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from plyer import filechooser
import seaborn as sns
import matplotlib.pyplot as plt
from kivymd.uix.dropdownitem import MDDropDownItem
from kivy.properties import ObjectProperty
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from matplotlib import rcParams
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn import svm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
class Content(MDFloatLayout):
    pass


class Image_btn(ButtonBehavior,Image):
    pass
class FirstWindow(Screen):
    theSelectedFile=dict()
    global thedataadd
    thedataadd=0

    def upload_menu(self):

        # to choose file by plyer lib
        self.files=filechooser.open_file(title="choose an excel files",filter=[('*.xlsx')],multiple=True)

        for file in self.files:
            file_name = file.split("\\")[-1]
            fileDF=pd.read_excel(file_name)
            box = MDBoxLayout(size_hint_y=.30, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.selecThefile)
            label = MDLabel(text=f"[color=#3f51b5]{file_name} \n"
                                 f" rows {fileDF.shape[0]} ,col {fileDF.shape[1]}"
                                 f" [/color]", markup=True, padding_x=20)
            box.add_widget(label)
            box.add_widget(checkbox)
            self.ids.yourfile.add_widget(box)
            self.theSelectedFile[checkbox] = file_name

    def selecThefile(self, checkbox, value):
        if value:
            global df
            df=pd.read_excel(self.theSelectedFile[checkbox])
            for x in df:
                box=MDBoxLayout(size_hint_y=None,height=50,padding=[30,0,0,0])
                colname = MDLabel(text=f"[color=#3f51b5]{x}[/color]",markup=True,padding_y=10,font_size="10sp")
                coltype = MDLabel(size_hint_x=.5,text=f"[color=#3f51b5]{df[x].dtype}[/color]", markup=True,font_size="10sp")
                box.add_widget(colname)
                box.add_widget(coltype)
                self.ids.fileinfo.add_widget(box)
            for x, y in self.theSelectedFile.items():
                if x !=checkbox:
                    global myRealDF
                    global myRealDFadd
                    myRealDF=pd.read_excel(y)
                    myRealDFadd = pd.read_excel(y)
                    global thedataadd
                    thedataadd=1




class SecondWindow(Screen):
    theValWanted = dict()
    theMissingColWanted = dict()
    theMissingCol=[]
    theOldVal=[]
    add=0
    mydf=0

    def on_enter(self, *args):
        parentBox = MDGridLayout(adaptive_height=True, cols=2)
        self.ids['theMissingCol'] = parentBox
        self.ids.missingbox.add_widget(parentBox)
        for col in df:
            if df[col].isnull().sum()>0:

                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                checkbox = MDCheckbox(size_hint_x=.25)
                checkbox.bind(active=self.selectTheMissingCol)
                themissing = MDLabel(text=f"[color=#3f51b5]{col} count: {df[col].isnull().sum()}[/color]", markup=True,
                              padding_y=10, font_size="10sp")
                box.add_widget(themissing)
                box.add_widget(checkbox)
                self.ids.theMissingCol.add_widget(box)
                self.theMissingColWanted[checkbox] = col
    def selectTheMissingCol(self,checkbox,value):
        if value:
            self.theMissingCol.append(self.theMissingColWanted[checkbox])
            if df[self.theMissingColWanted[checkbox]].dtypes=='object':
                self.ids.theNewVal.text=f"{df[self.theMissingColWanted[checkbox]].mode()[0]}"
            else:
                self.ids.theNewVal.text=f"{df[self.theMissingColWanted[checkbox]].mean()}"
        else:
            self.theMissingCol.remove(self.theMissingColWanted[checkbox])
            self.ids.theNewVal.text=""


    def changTheMissing(self):
        for col in self.theMissingCol:
            global df

            if df[col].isnull().sum()>0:
                if df[col].dtypes=='object':
                    thenewval = self.ids.theNewVal.text
                else:
                    thenewval=float(self.ids.theNewVal.text)
                df[col].fillna(thenewval,inplace=True)
                if thedataadd:
                    global myRealDF
                    myRealDF[col].fillna(thenewval, inplace=True)
                self.ids.theNewVal.text=''
        self.ids.missingbox.remove_widget(self.ids.theMissingCol)
        parentBox = MDGridLayout(adaptive_height=True, cols=2)
        self.ids['theMissingCol'] = parentBox
        self.ids.missingbox.add_widget(parentBox)
        for col in df:
            if df[col].isnull().sum() > 0:
                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                checkbox = MDCheckbox(size_hint_x=.25)
                checkbox.bind(active=self.selectTheMissingCol)
                themissing = MDLabel(text=f"[color=#3f51b5]{col} count: {df[col].isnull().sum()}[/color]", markup=True,
                                     padding_y=10, font_size="10sp")
                box.add_widget(themissing)
                box.add_widget(checkbox)
                self.ids.theMissingCol.add_widget(box)
                self.theMissingColWanted[checkbox] = col


    def dropRow(self):
        for col in self.theMissingCol:
            global df
            if df[col].isnull().sum()>0:
                df.dropna(subset=[f'{col}'],inplace=True)
                if thedataadd:
                    global myRealDF
                    myRealDF.dropna(subset=[f'{col}'], inplace=True)
        self.ids.missingbox.remove_widget(self.ids.theMissingCol)
        parentBox = MDGridLayout(adaptive_height=True, cols=2)
        self.ids['theMissingCol'] = parentBox
        self.ids.missingbox.add_widget(parentBox)
        for col in df:
            if df[col].isnull().sum() > 0:
                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                checkbox = MDCheckbox(size_hint_x=.25)
                checkbox.bind(active=self.selectTheMissingCol)
                themissing = MDLabel(text=f"[color=#3f51b5]{col} count: {df[col].isnull().sum()}[/color]", markup=True,
                                     padding_y=10, font_size="10sp")
                box.add_widget(themissing)
                box.add_widget(checkbox)
                self.ids.theMissingCol.add_widget(box)
                self.theMissingColWanted[checkbox] = col

    def dropcol(self):
        for col in self.theMissingCol:
            global df
            if df[col].isnull().sum() > 0:
                df.drop(col,axis=1,inplace=True)
                if thedataadd:
                    global myRealDF
                    myRealDF.drop(col, axis=1, inplace=True)
        self.ids.missingbox.remove_widget(self.ids.theMissingCol)
        parentBox = MDGridLayout(adaptive_height=True, cols=2)
        self.ids['theMissingCol'] = parentBox
        self.ids.missingbox.add_widget(parentBox)
        for col in df:
            if df[col].isnull().sum() > 0:
                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                checkbox = MDCheckbox(size_hint_x=.25)
                checkbox.bind(active=self.selectTheMissingCol)
                themissing = MDLabel(text=f"[color=#3f51b5]{col} count: {df[col].isnull().sum()}[/color]", markup=True,
                                     padding_y=10, font_size="10sp")
                box.add_widget(themissing)
                box.add_widget(checkbox)
                self.ids.theMissingCol.add_widget(box)
                self.theMissingColWanted[checkbox] = col

    def cat_dropdown(self):
        cat_col=[]
        for type in df.dtypes.index:
            if df.dtypes[type] =='object':
                cat_col.append(type)
        self.menu_catlist=[{
            "text":catcol,
            "viewclass":"OneLineListItem",
            "on_release":lambda x=catcol:self.valuesOfCat(x),

        }for catcol in cat_col
        ]
        self.catMenu =MDDropdownMenu(
            caller=self.ids.catmenu,
            items=self.menu_catlist,
            width_mult=4
        )
        if cat_col:
            self.catMenu.open()
    def valuesOfCat(self,catcol):
        if self.add:
            self.ids.catbox.remove_widget(self.ids.catValues)
        x=df[catcol].value_counts().to_frame()
        self.thecatcol=catcol
        self.catMenu.dismiss()
        parentBox=MDGridLayout(adaptive_height= True,cols=2)
        self.ids['catValues'] = parentBox
        self.ids.catbox.add_widget(parentBox)

        for i,ele in enumerate(x.values):
            self.add=1
            box=MDBoxLayout(size_hint_y=None,padding=[30,0,0,0])
            checkbox=MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.selectTheOLd)
            value = MDLabel(text=f"[color=#3f51b5]{x.index[i]} count: {x.values[i][0]}[/color]",markup=True,padding_y=10,font_size="10sp")
            box.add_widget(value)
            box.add_widget(checkbox)
            self.ids.catValues.add_widget(box)
            self.theValWanted[checkbox]=x.index[i]
    def selectTheOLd(self,checkbox,value):
        if value:
            self.theOldVal.append(self.theValWanted[checkbox])
        else:
            self.theOldVal.remove(self.theValWanted[checkbox])
    def changTheVal(self):

        for val in self.theOldVal:
            self.theNew=self.ids.theNewV.text
            df[self.thecatcol]=df[self.thecatcol].replace(val,self.theNew)
            if thedataadd:
                global myRealDF
                myRealDF[self.thecatcol] = myRealDF[self.thecatcol].replace(val, self.theNew)


class ThirdWindow(Screen):
    theExpColWanted = dict()
    theExpPlotWanted = dict()
    theExpcol=""

    plotadd=0
    plotcol=0
    color=sns.color_palette('colorblind')
    def on_enter(self, *args):
        parentBox = MDGridLayout(adaptive_height=True, cols=1)
        self.ids['theExpCol'] = parentBox
        self.ids.expcol.add_widget(parentBox)
        for col in df:
            box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.selectTheExpCol)
            theCol = MDLabel(text=f"[color=#3f51b5]{col}[/color]", markup=True, font_size="10sp")
            box.add_widget(theCol)
            box.add_widget(checkbox)
            self.ids.theExpCol.add_widget(box)
            self.theExpColWanted[checkbox] = col
    def selectTheExpCol(self,checkbox,value):
        if value:
            global df
            if self.plotcol:
                self.ids.expfig.remove_widget(self.ids.theExpPlot)

            self.plotcol=1
            self.theExpcol=self.theExpColWanted[checkbox]
            if df[self.theExpcol].dtypes == 'object':

                self.fig=sns.countplot(x=self.theExpcol ,data=df,palette=self.color)
                self.fig.set_xticklabels(self.fig.get_xticklabels(),rotation=90)
            else:
                self.fig=sns.displot(data=df, x=self.theExpcol, kde=True)

            self.theFig = FigureCanvasKivyAgg(plt.gcf())
            self.theFig.draw()
            self.ids['theExpPlot'] = self.theFig
            self.ids.theExpPlot.size=self.ids.expfig.size
            self.ids.expfig.add_widget(self.theFig)

    def savefig(self):
        print(self.theExpcol)
        plt.savefig(f"{self.theExpcol}.png")

class FourhtWindow(Screen):
    theTarColWanted = dict()
    theColCorrWanted = dict()
    corrAdd = 0
    plotCorrCol=0
    catCol=[]

    def on_enter(self, *args):
        global dataa
        dataa = df.copy()
        for col in df:
            if df[col].dtypes == 'object':
                self.catCol.append(col)
        if len(self.catCol)>0:
            dataa=pd.get_dummies(df,columns=self.catCol)
        parentBox = MDGridLayout(adaptive_height=True, cols=1)
        self.ids['theTarCorr'] = parentBox
        self.ids.tarCorr.add_widget(parentBox)
        for col in dataa:
            box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.selectTheTarCorr)
            theCol = MDLabel(text=f"[color=#3f51b5]{col}[/color]", markup=True, font_size="10sp")
            box.add_widget(theCol)
            box.add_widget(checkbox)
            self.ids.theTarCorr.add_widget(box)
            self.theTarColWanted[checkbox] = col
    def selectTheTarCorr(self,checkbox,value):
        if value:
            self.theColCorr=self.theTarColWanted[checkbox]
            global df
            if self.corrAdd:
                self.ids.colCorr.remove_widget(self.ids.theColCorr)
            self.corrAdd=1
            self.theTarCorr=self.theTarColWanted[checkbox]
            parentBox = MDGridLayout(adaptive_height=True, cols=1)
            self.ids['theColCorr'] = parentBox
            self.ids.colCorr.add_widget(parentBox)
            df_cor = dataa.corr()

            for x, y in df_cor[self.theTarCorr].items():
                if ((y > 0.7) | (y < -0.7)) & (x != self.theTarCorr):
                    box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                    checkbox = MDCheckbox(size_hint_x=.25)
                    checkbox.bind(active=self.plotTheCorrPlot)
                    theCol = MDLabel(text=f"[color=#3f51b5]{x}[/color]", markup=True, font_size="10sp")
                    box.add_widget(theCol)
                    box.add_widget(checkbox)
                    self.ids.theColCorr.add_widget(box)
                    self.theColCorrWanted[checkbox] = x

    def plotTheCorrPlot(self,checkbox,value):
        if value:
            global df
            self.theCorrCol = self.theColCorrWanted[checkbox]
            if self.plotCorrCol:
                self.ids.corrFig.remove_widget(self.ids.theCorrPlot)

            self.plotCorrCol = 1
            plt.clf()
            self.fig = sns.scatterplot(data=dataa, x=self.theTarCorr, y=self.theCorrCol)
            self.theCorrFig = FigureCanvasKivyAgg(plt.gcf())
            self.theCorrFig.draw()
            self.ids['theCorrPlot'] = self.theCorrFig
            self.ids.corrFig.add_widget(self.theCorrFig)
    def savefig(self):
        plt.savefig(f"{self.theTarCorr} and {self.theCorrCol}.png")
        print("done")

class FifthWindow(Screen):
    theModelWanted=dict()
    theDropCol=dict()
    catColend=[]
    drop=[]
    theTargetCol=''
    theTestAmount=''
    thereisCLustring=0
    global thedataadd
    def on_enter(self, *args):
        global data
        data = df.copy()
        if thedataadd:
            global myRealData
            global myRealDF
            myRealData=myRealDF.copy()
        self.model=['classification','regression','clustring']
        parentBox = MDGridLayout(adaptive_height=True, cols=1)
        self.ids['modelCheackbox'] = parentBox
        self.ids.theModelBox1.add_widget(parentBox)
        for m in self.model:
            box = MDBoxLayout(size_hint_y=.20, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.selecTheModel)
            theCol = MDLabel(text=f"[color=#3f51b5]{m}[/color]", markup=True, font_size="10sp")
            box.add_widget(theCol)
            box.add_widget(checkbox)
            self.ids.modelCheackbox.add_widget(box)
            self.theModelWanted[checkbox] = m
        parentBox = MDGridLayout(adaptive_height=True, cols=1)
        self.ids['dropbox'] = parentBox
        self.ids.dropCol.add_widget(parentBox)
        for col in data :
            box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.dropColumns)
            theCol = MDLabel(text=f"[color=#3f51b5]{col}[/color]", markup=True, font_size="10sp")
            box.add_widget(theCol)
            box.add_widget(checkbox)
            self.ids.dropbox.add_widget(box)
            self.theDropCol[checkbox] = col
    def dropColumns(self,checkbox,value):
        if value:
            self.drop.append(self.theDropCol[checkbox])

        else:
            self.drop.remove(self.theDropCol[checkbox])
    def dropColumnsBtn(self):
        global data
        for col in self.drop:
            data=data.drop(col,axis=1)
            if thedataadd:
                global myRealData
                myRealData = myRealData.drop(col, axis=1)
        self.drop=[]
        self.ids.dropCol.remove_widget(self.ids.dropbox)
        parentBox = MDGridLayout(adaptive_height=True, cols=1)
        self.ids['dropbox'] = parentBox
        self.ids.dropCol.add_widget(parentBox)
        for col in data:
            box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
            checkbox = MDCheckbox(size_hint_x=.25)
            checkbox.bind(active=self.dropColumns)
            theCol = MDLabel(text=f"[color=#3f51b5]{col}[/color]", markup=True, font_size="10sp")
            box.add_widget(theCol)
            box.add_widget(checkbox)
            self.ids.dropbox.add_widget(box)
            self.theDropCol[checkbox] = col


    def selecTheModel(self,checkbox,value):
        if value:
            global data
            for col in data:
                if data[col].dtypes == 'object' and col !=self.theTargetCol:
                    self.catColend.append(col)

            if len(self.catColend) > 0:
                data = pd.get_dummies(df, columns=self.catColend)
                if thedataadd:
                    global myRealData
                    myRealData = pd.get_dummies(myRealDF, columns=self.catColend)
            self.theSelectedModel = self.theModelWanted[checkbox]
            if self.theTargetCol:
                targetCol = self.theTargetCol
                if self.theSelectedModel == "clustring":
                    targetCol = int(targetCol)

            else:
                if self.theSelectedModel == "clustring":
                    targetCol = 3
                else:
                    for col in df.columns:
                        targetCol = col
            if self.theTestAmount:
                testAmount = self.theTestAmount
            else:
                testAmount = 0.3
            if self.theSelectedModel=="clustring":
                targetCol=float(targetCol)






            if self.theSelectedModel =="classification":
                self.X = data.drop(targetCol, axis=1).values
                self.Y = data[targetCol].values
                self.XReal = myRealData.drop(targetCol, axis=1).values
                self.XReal = preprocessing.StandardScaler().fit(self.XReal).transform(self.XReal.astype(float))
                classScore={"KNN":0,"Decision Trees":0,"Logistic Regression":0,"SVM":0}
                self.X = preprocessing.StandardScaler().fit(self.X).transform(self.X.astype(float))


                X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=testAmount, random_state=4)
                Ks = 10
                mean_acc = np.zeros((Ks - 1))
                std_acc = np.zeros((Ks - 1))

                for n in range(1, Ks):
                    # Train Model and Predict
                    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
                    yhat = neigh.predict(X_test)
                    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

                    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
                k=mean_acc.argmax()+1
                neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                yneigh = neigh.predict(X_test)
                classScore["KNN"]=metrics.accuracy_score(y_test, yneigh)
                drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
                drugTree.fit(X_train, y_train)
                yTree = drugTree.predict(X_test)
                classScore["Decision Trees"]=metrics.accuracy_score(y_test, yTree)
                LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
                yLR = LR.predict(X_test)
                classScore["Logistic Regression"] = metrics.accuracy_score(y_test, yLR)
                clf = svm.SVC(kernel='rbf')
                clf.fit(X_train, y_train)
                yclf = clf.predict(X_test)
                classScore["SVM"] = metrics.accuracy_score(y_test, yclf)
                theMaxScore=max(classScore,key=classScore.get)
                parentBox = MDGridLayout(adaptive_height=True, cols=1)
                self.ids['theModelValue'] = parentBox
                self.ids.theModelBox.add_widget(parentBox)
                for x, y in classScore.items():

                    box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                    theCol = MDLabel(text=f"[color=#3f51b5]the model {x} and the score is {y}[/color]", markup=True, font_size="10sp")
                    box.add_widget(theCol)
                    self.ids.theModelValue.add_widget(box)
                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                theCol = MDLabel(text=f"[color=#3f51b5]the best model is {theMaxScore}[/color]", markup=True,
                                 font_size="10sp")
                box.add_widget(theCol)
                self.ids.theModelValue.add_widget(box)
                if theMaxScore=='KNN':
                    ynew = neigh.predict(self.XReal)
                    myRealDFadd['the predicted value'] = ynew
                elif theMaxScore=='Decision Trees':
                    ynew = drugTree.predict(self.XReal)
                    myRealDFadd['the predicted value'] = ynew
                elif theMaxScore=='Logistic Regression':
                    ynew=LR.predict(self.XReal)
                    myRealDFadd['the predicted value'] =ynew
                else:
                    ynew= clf.predict(self.XReal)
                    myRealDFadd['the predicted value'] =ynew
                myRealDFadd.to_excel("classification.xlsx")

            elif self.theSelectedModel =="clustring":

                self.X = data.values
                self.X = preprocessing.StandardScaler().fit(self.X).transform(self.X.astype(float))
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42 )
                    kmeans.fit(self.X)
                    wcss.append(kmeans.inertia_)
                plt.clf()
                sns.set()
                plt.plot(range(1,11),wcss)
                self.theClusFig = FigureCanvasKivyAgg(plt.gcf())
                self.theClusFig.draw()
                self.ids.theModelBox.add_widget(self.theClusFig)
                self.thereisCLustring=1


            else:
                if thedataadd:
                    self.XReal = myRealData.drop(targetCol, axis=1).values
                    self.XReal = preprocessing.StandardScaler().fit(self.XReal).transform(self.XReal.astype(float))
                self.X = data.drop(targetCol, axis=1).values

                self.Y = data[targetCol].values
                self.X = preprocessing.StandardScaler().fit(self.X).transform(self.X.astype(float))

                X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=testAmount,random_state=4)
                regr = linear_model.LinearRegression()
                regr.fit(X_train, y_train)
                yMLR = regr.predict(X_test)
                ynew = regr.predict(self.XReal)
                myRealDFadd['the predicted value'] = ynew
                myRealDFadd.to_excel("regression.xlsx")
                regScore=metrics.r2_score(y_test,yMLR)
                regScore=abs(regScore*10)
                parentBox = MDGridLayout(adaptive_height=True, cols=1)
                self.ids['theModelValue'] = parentBox
                self.ids.theModelBox.add_widget(parentBox)
                box = MDBoxLayout(size_hint_y=None, padding=[30, 0, 100, 0])
                padding = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                padding1 = MDBoxLayout(size_hint_y=None, padding=[30, 0, 0, 0])
                theCol = MDLabel(text=f"[color=#3f51b5]the model Multiple Linear Regression and the score is {regScore}[/color]", markup=True,
                                 font_size="10sp")
                box.add_widget(theCol)
                self.ids.theModelValue.add_widget(box)
                self.ids.theModelValue.add_widget(padding)
                self.ids.theModelValue.add_widget(padding1)









    def selectTarget(self):
        if self.thereisCLustring:
            self.X = data.values
            self.X = preprocessing.StandardScaler().fit(self.X).transform(self.X.astype(float))
            n=int(self.ids.theTargetInput.text)
            kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42)
            labelKmeans=kmeans.fit_predict(self.X)
            df['the label'] = labelKmeans
            df.to_excel("clustring.xlsx")
            dbscan=DBSCAN(eps=n,min_samples=5)
            dbscan.fit(self.X)
            labelDB=dbscan.labels_
            core_samples=np.zeros_like(labelDB,dtype=bool)
            core_samples[dbscan.core_sample_indices_]=True



        self.theTargetCol=self.ids.theTargetInput.text


    def testAmount(self):
        self.theTestAmount = self.ids.theTestAmountInput.text
        self.theTestAmount=float(self.theTestAmount)

sm=ScreenManager()

sm.add_widget(FirstWindow(name='filescreen'))
sm.add_widget(SecondWindow(name='preprocessing'))
sm.add_widget(ThirdWindow(name='exploratory'))
sm.add_widget(FourhtWindow(name='corelation'))
sm.add_widget(FifthWindow(name='model'))

class MainApp(MDApp):
    def build(self):

        self.theme_cls.theme_style="Light"
        self.theme_cls.primary_palette="BlueGray"


        return Builder.load_file('mykivy.kv')





MainApp().run()