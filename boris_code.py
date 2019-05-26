from pyhive import presto
from getpass import getuser
from ba_infra.logger import WixLogger
import logging
import sqlparse
from pyhive.exc import DatabaseError
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *

logger = WixLogger(__name__, WixLogger.DEBUG)

class WixPrestoConnection:
    def __init__(self, host='presto.wixpress.com', port=8181, username=None):
        self.__host = host
        self.__port = port
        if username:
            self.__username = username
        elif getuser() == 'wix':
            self.__username = 'wix'
        else:
            self.__username = getuser() + '@wix.com'
        self._sql_length_limit = 900000  # max limit is 1,000,000 so leaving 100k as buffer
        self.__conn = presto.connect(host=self.__host, port=self.__port, username=self.__username)
        self.__cursor = self.__conn.cursor()


    def execute_sql(self, sql, parameters=None, rows_limit=None, add_cols_metadata=False):
        """
        Executes an sql query specified in `sql` param and returns the relevant data in a format of list of tuples. \n
        Supports every query supported by presto engine (select, create, drop, alter, etc.) \n
        :param sql: an sql command to run
        :param parameters: if the `sql` param contains a query with %s placeholder, \n
                a list of parameters can be passed in `parameters` param to fill those placeholders \n
                Example: \n
                `sql`: "select * from my_table where name=%s and category=%s" \n
                `parameters`: ['cars', 'sport'] \n
                The executed query will be: "select * from my_table where name='cars' and category='sport'" \n
        :param rows_limit: limits the number of rows to fetch from DB. default is unlimited. \n
        :param add_cols_metadata: if set to `True`, will add columns names and types to the return value. \n
                In this case the result will be a dict: \n
                {'rows': rows data, 'col_names': columns names, 'col_types': columns types} \n
        :param kwargs: can be any pair of optional key=value parameters `presto.Cursor.execute()` method supports \n
        :return: the result of the executed query as list of tuples. \n
        """
        self.__cursor.execute(sql, parameters=parameters)
        if not rows_limit:
            ret = self.__cursor.fetchall()
        else:
            ret = self.__cursor.fetchmany(rows_limit)
        if add_cols_metadata:
            names = [d[0] for d in self.__cursor.description]
            types = [d[1] for d in self.__cursor.description]
            ret = {'rows': ret, 'col_names': names, 'col_types': types}
        return ret

    def execute_sql_pandas(self, sql, parameters=None, rows_limit=None, parse_dates=False, **kwargs):
        """
        Executes an sql query specified in `sql` param and returns the relevant data in a format of pandas dataframe. \n
        Supports every query supported by presto engine (select, create, drop, alter, etc.) \n
        :param sql: an sql command to run
        :param parameters: if the `sql` param contains a query with %s placeholder, \n
                a list of parameters can be passed in `parameters` param to fill those placeholders \n
                Example: \n
                `sql`: "select * from my_table where name=%s and category=%s" \n
                `parameters`: ['cars', 'sport'] \n
                The executed query will be: "select * from my_table where name='cars' and category='sport'" \n
        :param rows_limit: limits the number of rows to fetch from DB. default is unlimited. \n
        :param parse_dates: by default date/timestamps are read as strings. \n
                if `parse_dates` is set to True, they will be read as pandas datetime64[ns] type \n
        :param kwargs: can be any pair of optional key=value parameters `pandas.DataFrame()` method supports \n
        :return: the result of the executed query as pandas dataframe. \n
        """
        ret = self.execute_sql(sql, parameters=parameters, rows_limit=rows_limit, add_cols_metadata=True)
        df = pd.DataFrame(ret['rows'], columns=ret['col_names'] if not kwargs.get('columns') else kwargs['columns'],
                          index=kwargs.get('index'), dtype=kwargs.get('dtype'), copy=kwargs.get('copy'))
        # convert date/timestamp columns to datetime dtype
        if parse_dates:
            for name, date_type in zip(ret['col_names'], ret['col_types']):
                if date_type in ('date', 'timestamp'):
                    df[name] = df[name].astype('datetime64[ns]')
        return df


    def execute_sql_spark(self, sql, spark:SparkSession, parameters=None, rows_limit=None, parallelize=None,
                          parse_dates=False, **kwargs):
        """
        Executes an sql query specified in `sql` param and returns the relevant data in a format spark dataframe. \n
        Supports every query supported by presto engine (select, create, drop, alter, etc.) \n
        :param sql: an sql command to run
        :param spark: a SparkSession object
        :param parameters: if the `sql` param contains a query with %s placeholder, \n
                a list of parameters can be passed in `parameters` param to fill those placeholders \n
                Example: \n
                `sql`: "select * from my_table where name=%s and category=%s" \n
                `parameters`: ['cars', 'sport'] \n
                The executed query will be: "select * from my_table where name='cars' and category='sport'" \n
        :param rows_limit: limits the number of rows to fetch from DB. default is unlimited. \n
        :param parallelize: defines to how many tasks divide the loading of data into spark DataFrame \n
        :param parse_dates: by default, presto columns of date/timestamp types are read as spark StringType. \n
                if `parse_dates` set to True, the columns will be parsed to spark DateType/TimestampType. \n
        :param kwargs: can be any pair of optional key=value parameters `pyspark.sql.SparkSession.createDataFrame()` method supports \n
        :return: the result of the executed query as a spark dataframe. \n
        """
        def generate_schema(col_names, col_types, parse_dates=False):
            return StructType([StructField(n, self._presto_type_to_spark(t, parse_dates))
                               for n, t in zip(col_names, col_types)])

        res = self.execute_sql(sql, parameters=parameters, rows_limit=rows_limit, add_cols_metadata=True)
        # if parallelize is not specified, then divide the number of rows by 100, but not less than 1 or more than 1000
        parallelize = min(1000, max(1, int(len(res['rows'])/ 100))) if not parallelize else parallelize

        # if schema wasn't passed generate schema based on results metadata
        if not kwargs.get('schema'):
            schema = generate_schema(res['col_names'], res['col_types'], parse_dates)
        else:
            schema = kwargs.get('schema')
            del kwargs['schema']

        # parse date/timestamp columns from string format
        if parse_dates and len(res['rows']) > 0:
            # load the date into spark temp view
            spark.createDataFrame(spark.sparkContext.parallelize(res['rows'], parallelize), schema=schema.names, **kwargs)\
                .createOrReplaceTempView('__temp__')
            # create df by casting date/timestamp columns
            df = spark.sql('select {} from __temp__'
                           .format(','.join([('cast({} as {})' if t in ('date', 'timestamp') else '{}').format(n, t)
                                             for n,t in zip(schema.names, res['col_types'])])))
            if df.schema != schema: # if the result schema is differs from the one defined, change it
                df = spark.createDataFrame(df.rdd, schema)
        else:
            df = spark.createDataFrame(spark.sparkContext.parallelize(res['rows'], parallelize), schema=schema, **kwargs)
        return df


    def _generate_values(self, rows, col_types):
        """
        creates a 'VALUES' statement string from input rows values.
        The pattern will be 'VALUES ((rows[0][0], ..., rows[0][n]), ..., (rows[m][0], ..., rows[m][n]))'
        :param rows: data set of rows represented by list of list, list of tuples or tuples of tuples
        :param col_types: supported types: [varchar, bigint, integer, decimal, double, \n
                            timestamp, date, boolean, array<type>, map<type,type>, expression].
        :return:
        """

        def generate_column_string(col, type):
            # convert types in the format of varchar(10)/numeric(5,2) => varchar/numeric
            type = type.split('(')[0].lower()
            # col = None if col is None else str(col) # as long as the col value isn't None(null) convert it to string
            if col is None or str(col).lower() == 'null' or str(col).lower() == 'nan' and type != 'expression':
                return 'cast(null as {})'.format(type)
            elif type == 'varchar':
                return "cast('" + str(col).replace("'", "''") + "' as varchar)"
            elif type in ['bigint', 'integer', 'decimal', 'double', 'boolean']:
                return str(col)
            elif type == 'timestamp':
                return "timestamp '" + str(col) + "'"
            elif type == 'date':
                return "date '" + str(col) + "'"
            elif type == 'expression':
                return str(col)
            elif type == 'array':
                return 'array' + str(col)
            elif 'array' in type:
                return 'cast(array{} as {})'.format(str(col), type)
            elif 'map' in type:
                return 'map(array{},array{})'.format(list(dict(col).keys()), list(dict(col).values()))
            else:
                logger.error('Unknown column type: ' + str(type))
                pass

        sql = ''
        rows_handled = 0
        for row in rows:
            value = '('
            for i, col in enumerate(row):
                col_type = 'varchar' if col_types is None else col_types[i]
                value += generate_column_string(col, col_type) + ','
            value = value[:-1] + ')'
            sql += value + ','
            rows_handled += 1
            if len(sql) > self._sql_length_limit:
                break

        sql = sql[:-1]
        rows = rows[rows_handled:]
        sql = 'VALUES ' + sql
        return (sql, rows)

    def _presto_type_to_spark(self, date_type:str, parse_date=False):
        if date_type  == 'integer':
            return IntegerType()
        elif date_type == 'boolean':
            return BooleanType()
        elif date_type == 'bigint':
            return LongType()
        elif date_type == 'double' or date_type.startswith('decimal'):
            return FloatType()
        elif date_type.startswith('varchar'):
            return StringType()
        elif date_type == 'date':
            return DateType() if parse_date else StringType()
        elif date_type == 'timestamp':
            return TimestampType() if parse_date else StringType()
        elif date_type.startswith('array'):
            # if array type specified i.e. array(bigint)
            if len(date_type.split('(')) > 1:
                return ArrayType(self._presto_type_to_spark(date_type.split('(')[1]))
            else:
                return ArrayType(StringType())
        elif date_type.startswith('map'):
            return MapType(StringType(), StringType())

    def _spark_type_to_presto(self, data_type):
        mapping = {'BooleanType':'boolean',
                   'DateType': 'date',
                   'DoubleType': 'double',
                   'IntegerType': 'integer',
                   'LongType': 'bigint',
                   'ShortType': 'integer',
                   'StringType': 'varchar',
                   'TimestampType': 'timestamp'}
        return mapping.get(str(data_type))

    def _pandas_type_to_presto(self, data_type):
        data_type = ''.join([c for c in str(data_type) if not c.isdigit()])
        mapping = {'object': 'varchar',
                   'int': 'bigint',
                   'uint': 'bigint',
                   'float':'double',
                   'bool': 'boolean',
                   'datetime[ns]': 'timestamp'}
        return mapping.get(data_type)

    def write_data(self, table_name, data, col_types: list = None, col_names: list = None, col_dict: dict = None, if_exists='fail'):
        """
        Writes data into a table. \n
        :type table_name: str
        :param table_name: a full presto table name in format of [db].[schema].[table_name] \n
        :type data: list[list] | tuple[tuple] | list[tuple] | tuple[list] | spark DataFrame
        :param data: a data set representing rows in a table. \n
                supported formats: list/tuple of lists/tuples, spark DataFrame, pandas DataFrame \n
        :type col_types: list[str]
        :param col_types: list of strings that defines the columns types in the table. \n
                must be passed if the target table doesn't exist. \n
                supported types: [varchar, bigint, integer, decimal, double, timestamp, date, boolean, array<type>, map<type,type>, expression] \n
                expression data type: a special type that can be specified when passing a string that represent an sql expression \n
                example of possible expression: substr('some string', 1, 5)
        :type col_names: list[str]
        :param col_names: list that defines the column names in the table. must be passed if table doesn't exists.
        :type col_dict: dict
        :param col_dict: a dict defines column names and types. both keys and values should be strings. \n
                can be passed instead of col_names AND col_types. \n
                example: {"name":"varchar", "age":"integer", "birth_date":"date"}
        :type if_exists: str
        :param if_exists: sets the behavior of the method if the table already exists: \n
            'fail' - the method will throw exception if the table already exists \n
            'append' - will insert the data if the table already exists (no need to pass col_types/col_names)
                or create the table if it doesn't exists (in this case col_types and col_names must be passed) \n
            'replace' - will drop the old table and create a new one instead. if col_types and col_names aren't passed,
                the new table will be in the same structure as the old one, otherwise it will be based on col_types/col_names
        :return: the number of rows inserted
        """
        #TODO: support writing of pandas Dataframes
        metadata = self.get_table_metadata(table_name)

        # table exists and if_exists set to fail
        if if_exists.lower() == 'fail' and metadata:
            raise Exception('Table {} already exists'.format(table_name))

        # if table doesn't exist but col_types/col_names parameter is empty
        if not metadata and not ((col_names and col_types) or col_dict):
            # if data is a spark DataFrame object, get columns names and types from its schema
            if isinstance(data, DataFrame):
                col_names = data.schema.names
                col_types = [self._spark_type_to_presto(field.dataType) for field in data.schema.fields]
            # if data is a pandas DataFrame object, get columns names and types from its schema
            elif isinstance(data, pd.DataFrame):
                col_names = data.columns.values.tolist()
                col_types = [self._pandas_type_to_presto(dt) for dt in data.dtypes]
            else:
                raise Exception('Table {} does not exists, please specify both `col_names` and `col_types` parameters'
                            .format(table_name))

        # if table exists and if_exists set to replace mode: drop old table
        if if_exists.lower() == 'replace' and metadata:
            self.execute_sql('DROP TABLE IF EXISTS {}'.format(table_name))

        # if column types/names were passed use it, else use col_dict, else use the types/names from table metadata
        col_types = col_types if col_types else list(col_dict.values()) if col_dict else list(metadata.values())
        col_names = ', '.join(col_names if col_names else list(col_dict.keys()) if col_dict else list(metadata.keys()))
        rows_inserted = 0

        # if data is a spark DataFrame object, collect the data and convert to list of tuples format
        if isinstance(data, DataFrame):
            data = [tuple(row) for row in data.collect()]

        # if data is a pandas DataFrame object, collect the data and convert to list of tuples format
        if isinstance(data, pd.DataFrame):
            data = data.values.tolist()

        # if table doesn't exist or if_exists set to replace mode: create the table
        if not metadata or if_exists.lower() == 'replace':
            result = self._generate_values(rows=data, col_types=col_types)
            sql = 'CREATE TABLE {table} AS SELECT * FROM ({values}) t({cols})' \
                .format(table=table_name, values=result[0], cols=col_names)
            self.execute_sql(sql)
            rows_inserted += len(data) - len(result[1])
            data = result[1]

        # append data if any remains
        while data:
            result = self._generate_values(rows=data, col_types=col_types)
            sql = 'INSERT INTO {table} {values}'.format(table=table_name, values=result[0])
            self.execute_sql(sql)
            rows_inserted += len(data) - len(result[1])
            data = result[1]

        return rows_inserted

    def execute_batch_sql(self, batch_sql: str):
        """
        The method executes multiple sql statements one after another passed in `batch_sql` param. \n
        The sql statements should be separated by ';'. \n
        Comments are ignored. \n
        :param batch_sql: sql script in string format \n
        :return: on success the method will return the dict: {'succeeded': True}. \n
                on failure the method will return the dict: \n
                {'succeeded': False, \n
                'error': <a string containing the error that caused the last sql statement to fail>, \n
                'sqls': <a list of the remaining sql statement that weren't executed yet> } \n
        """
        # removes comments from batch_sql and splits into single sql statements. then removes empty statements
        sql_statements = list(filter(None, sqlparse.format(batch_sql, strip_comments=True).split(';')))

        # try to execute the sql statements
        try:
            while sql_statements:
                sql = sql_statements[0].strip()
                if len(sql) > 0:
                    logging.debug('SQL statement to execute:\n {}'.format(sql))
                    res = self.execute_sql(sql)
                    logging.debug('SQL statement completed with the result: \n {}'.format(res))
                sql_statements.pop(0)
            return {'succeeded': True}
        except Exception as e:
            logging.error('SQL statement failed with the error:\n {}'.format(str(e)))
            return {'succeeded': False, 'error': str(e), 'sqls': sql_statements}

    def update_table(self, table_name: str, set_clause: dict, where_clause: str = None):
        """
        allows to update rows in a table \n
        :param table_name: full table name including catalog and schema, e.i. qbox.ba_guild.my_table \n
        :param set_clause: a set clause in a dict format. \n
                the keys are the column names and values are the values to set into the columns. \n
                all values should be in string format and can include presto expressions. \n
                example: {'col1' : 'cast(col1 as date)',
                        'col2' : "'my string'", \n
                        'col2  : "date '2018-08-08'" } \n
        :param where_clause: a string, containing a where clause statement to specify the rows to update. \n
                the statement shouldn't contains preceding 'where'. \n
                example: "col1 = 'Test Case' and col2 < date '2018-01-01' and (col2 > 4 or col2 < 1)" \n
                if `where_clause` isn't specified, the whole table will be updated. \n
        :return: number of rows that were updated
        """
        # if no where clause defined make it 1=1
        where_clause = '1=1' if where_clause is None else where_clause

        # get the table columns' names
        metadata = self.execute_sql('describe {}'.format(table_name))

        # replace the names of columns that were set for update with their new value. m[0] is the column name
        cols = [set_clause.get(m[0]) + ' as ' + m[0] if set_clause.get(m[0]) else m[0] for m in metadata]

        # union between unchanged data and updated data
        sql = 'select count(1) from {table} where {where}'.format(table=table_name, where=where_clause)
        num_rows_updated = self.execute_sql(sql)[0][0]

        sql = 'create table {table}___updated as ' \
              'select {cols} from {table} ' \
              'where {where} ' \
              'union all ' \
              'select * from {table} ' \
              'where not ({where})' \
            .format(cols=', '.join(cols), table=table_name, where=where_clause)
        self.execute_sql(sql)

        # swap old table with updated and drop old table
        self.swap_tables(table_name + '___updated', table_name, keep_backup=False)

        return num_rows_updated

    def get_table_metadata(self, table_name):
        """
        if the table exists, will return its column names and types as a dict, otherwise will return `None` \n
        can be used to check if a table exits. \n
        :param table_name: full table name - [db].[schema].[name] \n
        :return: None if table doesn't exist else a dict containing column names and types
        """
        try:
            ret = self.execute_sql('select * from {} limit 1'.format(table_name), add_cols_metadata=True)
            return dict(zip(ret['col_names'], ret['col_types']))
        except DatabaseError as e:
            if 'Table {} does not exist'.format(table_name) in e.args[0]['message']:
                return None
            else:
                raise

    def swap_tables(self, new_table, old_table, ignore_if_not_exists=False, keep_backup=False):
        """
        Drops the `old_table` and renames `new_table` to `old_table`. \n
        If `old_table` doesn't exists the process will fail unles `ignore_if_not_exists=True` is specified. \n
        If `keep_backup=True` is specified a backup of the `old_table` will be kept with `___old` suffix. \n
        :param new_table: full new table name, including schema \n
        :param old_table: full old table name, including schema \n
        :param ignore_if_not_exists: if set to `True`, the process won't fail if 'old_table' does not exists. \n
        :param keep_backup: if set to `True` will keep the `old_table` as back up with `___old` suffix. default is `False`. \n
        :return:
        """
        drop_template = 'DROP TABLE IF EXISTS {0}'
        alter_template = 'ALTER TABLE {0} RENAME TO {1}'
        backup_table = old_table + '___old'
        try:
            self.execute_sql(drop_template.format(backup_table))
            try:
                self.execute_sql(alter_template.format(old_table, backup_table))
            except:
                if ignore_if_not_exists:
                    pass
                else:
                    raise
            self.execute_sql(alter_template.format(new_table, old_table))
            if not keep_backup:
                self.execute_sql(drop_template.format(backup_table))
        except Exception as e:
            logger.error('Failed to swap tables:' + str(e))
            raise e
        return True



    def recreate_table(self, table_name):
        def retriable_error(e):
            # query running time exceeded limit
            if 'exceeded' in str(e).lower():
                return False
            # query queue is full
            if 'too many queued queries' in str(e).lower():
                return False
            return True

        # drop table_name_new
        try:
            logger.debug('Trying to drop {}___new'.format(table_name))
            self.execute_sql('DROP TABLE IF EXISTS {}___new'.format(table_name))
            logger.debug('Successfully dropped {}___new'.format(table_name))
        except Exception as e:
            logger.error('Failed to drop {}___new'.format(table_name))
            logger.error('ERROR: ' + str(e))
            raise e

        # recreate table_name_new from table_name, with 5 retries
        MAX_RETRIES = 5
        retries = 0
        creation_succeeded = False
        while not creation_succeeded:
            try:
                logger.debug('Trying to create {0}___new from {0}'.format(table_name))
                self.execute_sql('CREATE TABLE {0}___new AS SELECT * FROM {0}'.format(table_name))
                logger.debug('Successfully created {}___new'.format(table_name))
                creation_succeeded = True
            except Exception as e:
                # if max attempts not yet reached, and the error should be retried
                if retries < MAX_RETRIES and retriable_error(e):
                    retries += 1
                    logger.warning('Failed to create {0}___new from {0}: '.format(table_name) + str(e))
                    logger.warning('Retrying again {0}/{1}'.format(retries, MAX_RETRIES))
                else:
                    logger.error('Failed to create {0}___new from {0}: '.format(table_name) + str(e))
                    logger.error("Max number of retries ({}) reached or the occurred error shouldn't be retried!"
                                 .format(str(MAX_RETRIES)))
                    raise e

        try:
            # drop table_name_old
            logger.debug('Trying to drop {}___old'.format(table_name))
            self.execute_sql('DROP TABLE IF EXISTS {}___old'.format(table_name))
            logger.debug('Successfully dropped {}___old'.format(table_name))
            # alter table_name to table_name_old
            logger.debug('Trying to rename {0} to {0}___old'.format(table_name))
            self.execute_sql('ALTER TABLE {0} RENAME TO {0}___old'.format(table_name))
            logger.debug('Successfully renamed {0} to {0}___old'.format(table_name))
            # alter table_name_new to table_name
            logger.debug('Trying to rename {0}___new to {0}'.format(table_name))
            self.execute_sql('ALTER TABLE {0}___new RENAME TO {0}'.format(table_name))
            logger.debug('Successfully renamed {0}___new to {0}'.format(table_name))
        except Exception as e:
            logger.error('Failed to perform operation:'.format(table_name) + str(e))
            raise e

        return True

    #TODO: implement sql execute that returns as spark data frame





    ######## Deprecated methods
    def bulk_insert(self, table_name, rows, col_types=None):
        """
        Deprecated. please use write_date method instead.
        """
        insert_template = 'INSERT INTO {} '.format(table_name)
        rows_inserted = 0
        while rows != []:
            res = self._generate_values(rows=rows, col_types=col_types)
            sql = insert_template + res[0]
            rows_inserted += len(rows) - len(res[1])
            rows = res[1]
            self.execute_sql(sql)
        return rows_inserted

    def create_from_data(self, table_name, rows, col_names=None, col_types=None, drop_if_exists=False, col_dict=None):
        """
        Deprecated. please use write_date method instead.
        :return:
        """
        # check args validity:
        if not col_names and not col_dict:
            logger.error('Some args are missing. col_names or col_dict must not be None')
            raise Exception('Some args are missing. col_names or col_dict must not be None')

        # if a col_dict is passed (with col names and their types)  fill col_names, col_types from it
        if col_dict:
            col_names = list(dict(col_dict).keys())
            col_types = list(dict(col_dict).values())

        create_template = 'CREATE TABLE {table} AS SELECT * FROM ({values}) t({cols})'
        drop_template = 'DROP TABLE IF EXISTS {}'
        col_names = ', '.join(col_names)

        # drop destination table if exists
        if drop_if_exists:
            self.execute_sql(drop_template.format(table_name))

        values_list = []
        while rows != []:
            res = self._generate_values(rows=rows, col_types=col_types)
            values_list.append(res[0])
            rows = res[1]

        # if create could be done in one statement (the query limit isn't reached)
        if len(values_list) == 1:
            sql = create_template.format(table=table_name, values=values_list[0], cols=col_names)
            self.execute_sql(sql)
        else:
            tmp_tables_list = []
            for i, values in enumerate(values_list):
                tmp_table_name = table_name + '___' + str(i)
                self.execute_sql(drop_template.format(tmp_table_name))
                sql = create_template.format(table=tmp_table_name, values=values, cols=col_names)
                self.execute_sql(sql)
                tmp_tables_list.append(tmp_table_name)
            # create as statm
            sql = 'CREATE TABLE {table} AS ({union})' \
                .format(table=table_name, union=' union '.join(['select * from ' + name for name in tmp_tables_list]))
            self.execute_sql(sql)
            for tmp_table_name in tmp_tables_list:
                self.execute_sql(drop_template.format(tmp_table_name))
        return True
