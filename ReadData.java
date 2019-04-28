package ReadData;

import java.io.FileReader;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.util.*;

public class ReadData {
    public static void main(String[] args){
        try{
            HashMap<String,HashMap<String, String[]>> tyIdMap=new HashMap<>();
            HashMap<String,String[]> timeMap=new HashMap<>();
            LineNumberReader reader=new LineNumberReader(new FileReader("C:\\Users\\AIR\\Desktop\\OutputData\\Research\\预报中心data\\OutputAsTxt.txt"));
            String[] strings;
            String dataString;
            for (String string:reader.readLine().split("\t")
                 ) {
                System.out.println(string);
            }
            System.out.println(reader.readLine().split("\t").toString());
        while((dataString=reader.readLine())!=null){
            strings=dataString.split("\t");
            if(strings.length<7)continue;
            if(!tyIdMap.containsKey(strings[0])){
                tyIdMap.put(strings[0],new HashMap<String,String[]>());
            }
            if(tyIdMap.containsKey(strings[0])){
                if(!tyIdMap.get(strings[0]).containsKey(strings[3])){
                    tyIdMap.get(strings[0]).put(strings[3],new String[5]);
                }
                if(tyIdMap.get(strings[0]).containsKey(strings[3])){
                    switch (strings[2]){
                        case "BABJ":
                            tyIdMap.get(strings[0]).get(strings[3])[0]=strings[6];
                            break;
                        case "KSLR":
                            tyIdMap.get(strings[0]).get(strings[3])[1]=strings[6];
                            break;
                        case "VHHH":
                            tyIdMap.get(strings[0]).get(strings[3])[2]=strings[6];
                            break;
                        case "RJTD":
                            tyIdMap.get(strings[0]).get(strings[3])[3]=strings[6];
                            break;
                        case "PGTW":
                            tyIdMap.get(strings[0]).get(strings[3])[4]=strings[6];
                            break;
                        default:
                            break;
                    }
                }

            }

        }
            PrintWriter writer=new PrintWriter("C:\\Users\\AIR\\Desktop\\OutputData\\Research\\预报中心data\\FormatAsTxt.txt");
        System.out.println(tyIdMap.toString());
        Set<String> keySet=tyIdMap.keySet();
        for (String string:keySet) {
            timeMap=tyIdMap.get(string);
            if(timeMap==null)continue;
            Set<String> timeSet=timeMap.keySet();
            for (String timeString:timeSet) {
                String[] dataList=timeMap.get(timeString);
                //Arrays.fill(dataList,"0");
                if(dataList==null||dataList.length<5)continue;
                if(dataList[0]==null||dataList[1]==null||dataList[2]==null||dataList[3]==null||dataList[4]==null)continue;
                if(new Integer(dataList[0])<=0||new Integer(dataList[1])<=0||new Integer(dataList[2])<=0||new Integer(dataList[3])<=0)continue;
                String output=string+"\t"+timeString+"\t"+dataList[0]+"\t"+ dataList[1]+"\t"+dataList[2]+"\t"+dataList[3]+"\t"+dataList[4]+"\n";
                writer.print(output);
            }
        }
        }catch (Exception e){
            e.printStackTrace();
            System.out.println(e);
        }
    }

}
