import sqlite3

import pandas as pd


class RecdataDB(object):

    @staticmethod
    def excel_to_db(excel_path):
        with pd.ExcelFile(excel_path) as xls:
            df = pd.read_excel(xls)
        columns = df.columns
        con = sqlite3.connect('test.db')
        cur = con.cursor()
        create_table_expr = f'''
            CREATE TABLE 电力用户表
            ({columns[0]} text, {columns[1]} text, {columns[2]} text)
        '''
        cur.execute(create_table_expr)
        # ['a', 'b', 'c'] -> '"a", "b", "c"'
        def expr(df_row):
            expr = ''
            for i, elem in enumerate(df_row):
                if i == len(df_row) - 1:
                    expr += f'"{elem}"'
                else:
                    expr += f'"{elem}",'
            expr = f'({expr})'

            return expr
        insert_expr = lambda df_row: f'''
            INSERT INTO 电力用户表 VALUES
            {expr(df_row)}
        '''
        insert_expr1 = insert_expr(df.loc[0])
        insert_expr2 = insert_expr(df.loc[1])
        cur.execute(insert_expr1)
        cur.execute(insert_expr2)
        con.commit()
        con.close()

        return df

