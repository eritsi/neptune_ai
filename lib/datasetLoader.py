#!/usr/bin/env python3
# coding: utf-8

import os
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

'''
BQデータテーブル読み込み処理プログラム
'''
class datasetLoader( object ):

    # GCPの設定を行う
    def __init__( self ):    
        self.project = os.environ["GCLOUD_PROJECT"]
        self.bqclient = bigquery.Client(project=self.project, location="asia-northeast1")
        self.bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        return

    # ファイルパスから全文を読む
    def __filePathToLines( self, in_FilePath ):
        lines = None
        with open( in_FilePath, "r" ) as _handle:
            lines = _handle.read()
            _handle.close()
        return lines

    # SQLを実行し、データフレームへ入れる
    def __sqlToDataframe( self, in_Lines ):

        # read bq table through bqstorage_client
        df = (
            self.bqclient.query(in_Lines)
            .result()
            .to_dataframe(
                bqstorage_client=self.bqstorageclient
            )
        )
        return df
    
    # データ読み込み処理
    def load_by_file( self, in_FilePath ):
        """SQLファイルを実行し、dfに入れる
        Parameters
        ----------
        in_FilePath : ファイルパス
        
        Returns
        -------
        df : pandasのdataframeが返る

        Examples
        --------
        >>> import datasetLoader
        >>> dataset_loader = datasetLoader()
        >>> df = dataset_loader.load_by_file( "./test.sql" )
        """
        whole_dataset = []
        lines = self.__filePathToLines( in_FilePath )
        whole_dataset  = self.__sqlToDataframe( lines )

        return whole_dataset

    # データ読み込み処理
    def load( self, lines ):
        """SQLを実行し、dfに入れる
        Parameters
        ----------
        lines : SQL文
        
        Returns
        -------
        df : pandasのdataframeが返る
        
        Examples
        --------
        >>> import datasetLoader
        >>> dataset_loader = datasetLoader()
        >>> sql = '''
            SELECT
              *
            FROM
              `bigquery-public-data.baseball.schedules`
          '''
        >>> df = dataset_loader.load( sql )
        """
        whole_dataset = []
        whole_dataset  = self.__sqlToDataframe( lines )

        return whole_dataset    