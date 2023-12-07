import re
evolution_dict = {
    # BGL
    "idoproxydb hit assert condition assert expression = <*> source file = idotransportmgr cpp source line = <*> function = int ido transport mgr send packet ido udp mgr* bgl ctl pav trace*":
    "idoproxydb hit assert",

    "data tlb error interrupt":
    "data tlb has an error",

    "program interrupt <*> <*>":
    "program report interrupt at <*>",

    "instruction <*> <*>":
    "command <*> <*>",

    "machine check <*>":
    "machine self check <*>",

    "data address <*>":
    "data memory address <*>",

    "data storage interrupt":
    "data storage interrupt when execute <*>",

    "floating pt ex mode <*> <*>":
    "logging : floating pt ex mode <*> <*>",

    "debug <*> <*>":
    "system debug <*> <*>",

    "data store interrupt caused by <*>":
    "data store interrupt",

    "machine state register <*>":
    "system state register <*>",

    "wait state <*>":
    "logging : wait state <*>",

    "critical input interrupt <*>":
    "critical input interrupt <*> at <*> in <*>",

    "problem state <*> = <*> = usr <*>":
    "problem state <*> equal <*> equal usr <*>",

    "node card is not fully functional":
    "node card has something wrong",

    "can not get assembly information for node card":
    "error when getting assembly information for node card",

    "ido packet timeout":
    "ido packet too long",

    # HDFS
    "receiving block (blk) src : <*> : <*> dest : <*> :  50010":
    "receiving block (blk) from src : <*> : <*> dest : <*> :  50010",

    "block*  name system.add stored block : block map updated : <*> :  50010 is added to (blk) size <*>":
    "block*  name system.add stored block : block map updated : add to (blk) <*> :  50010 , size <*>",

    "packet responder <*> for block (blk) <*>":
    "packet responder <*> for (BLK)",

    "received block (blk) of size <*> from <*>":
    "received block (blk) of size <*> from block <*>",

    "deleting block (blk) file <*>":
    "deleting (blk) <*>",

    "block*  name system.delete : (blk) is added to invalid set of <*> :  50010":
    "system.delete (blk) is added to invalid set for port 50010",

    "block*  name system.allocate block : <*> (blk)":
    "block*  name system allocate block : <*> (blk)",

    "<*> :  50010  served block (blk) to <*>":
    "<*> :  served block (blk)",

    "<*> :  50010 :  got exception while serving (blk) to <*> :":
    "<*> :  50010 :  got exception when serving (blk) to <*> :",

    "verification succeeded for (blk)":
    "succeeded verification for (blk)",
    


}

evolution_info = {
    "in_dict": 0,
    "not_in_dict":0
}

camel_sub = re.compile(r"((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\\-\\_])[A-Za-z]|[\\-\\_])")
def camel_segment(s:str) -> str:
    if s.islower():
        return s
    if re.match(r"[A-Z]+(?:s|es)", s):
        return s.lower()
    return camel_sub.sub(r' \1', s).lower()


def evolution_template(template:str):
    # template = camel_segment(template)
    if template in evolution_dict:
        evolution_info['in_dict'] += 1
        return evolution_dict[template]
        # return "this is a template"
    else:
        evolution_info['not_in_dict'] += 1
        tokens = template.split()
        return " ".join(tokens[:-1])
        # return " ".join(tokens[:int(len(tokens))])
