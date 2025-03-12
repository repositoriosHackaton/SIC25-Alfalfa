import load_data as ldata
def limpieza(df, df2, df3, df4):
    import pandas as pd
    import re
    
    #Eliminar filas en blanco
    df_dropped = df.dropna().sum()
    df2_dropped = df2.dropna().sum()
    print(df_dropped, df2_dropped)
    

    #Verificar duplicados
    df_duplicated = df.duplicated().sum()
    df2_duplicated = df2.duplicated().sum()
    print(df_duplicated, df2_duplicated)
    
    #Eliminar duplicados 
    df = df.drop_duplicates()
    df2 = df2.drop_duplicates()
    df4 = df4.drop_duplicates()
    df.duplicated().sum()
    df2.duplicated().sum()
    
    #Renombrar columnas
    df.rename(columns={"Article":"text"}, inplace=True)
    df2.rename(columns={"CleanText":"text"}, inplace= True)
    df4.rename(columns={"Content":"text"}, inplace= True)

    #eliminar columnas innecearias
    df = df.drop(columns=["Title", "Link", "Label"])
    df2 = df2.drop(columns=["RawText", "TTP", "URL"])
    df4 = df4.drop(columns=["labels"])
    print(df,df2)
    
    #Unir dfs
    Data_clean=pd.concat([df,df2,df3,df4],axis=0)
    Data_clean =Data_clean.drop(columns=["Unnamed: 0"])

    #eliminar caracteres invisibles
    def eliminar_caracteres_especiales(texto):
        return re.sub(r'[^a-zA-Z0-9\s]', '',texto)
    Data_clean['text'] = Data_clean['text'].apply(eliminar_caracteres_especiales)
    Data_clean.to_csv("datasets/clean_data.csv", index=False)
    #regresa la data a la funcion que la necesita
    print(Data_clean.count())
    return Data_clean

    
limpieza(ldata.ldata1(), ldata.ldata2(), ldata.ldata3(), ldata.ldata4())

