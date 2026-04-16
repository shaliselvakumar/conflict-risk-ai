import pandas as pd

def load_data():

    data = [
        ["Ukraine",30,"war escalation"],
        ["Gaza",40,"bombing crisis"],
        ["Sudan",28,"civil war"],
        ["Syria",20,"conflict"],
        ["Yemen",18,"airstrikes"],
        ["Afghanistan",15,"militant clashes"],
        ["Iran",32,"missile tensions"],
        ["Israel",35,"war escalation"],
        ["Pakistan",17,"border conflict"],
        ["Ethiopia",19,"regional clashes"],
        ["Myanmar",25,"civil unrest"],
        ["Nigeria",21,"terror attacks"],
        ["Mali",23,"insurgency"],
        ["Somalia",27,"extremism"],
        ["DR Congo",29,"armed conflict"],
        ["India",10,"border tension"],
        ["China",8,"geopolitical tension"],
        ["Russia",12,"military activity"],
        ["USA",5,"stable"],
        ["UK",4,"stable"],
        ["France",6,"low unrest"],
        ["Germany",3,"stable"],
        ["UAE",5,"stable but strategic"],
        ["Saudi Arabia",9,"regional influence"],
        ["Turkey",13,"regional tension"]
    ]

    return pd.DataFrame(data, columns=["location","event_intensity","text"])