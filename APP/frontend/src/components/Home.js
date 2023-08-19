import React, {useState} from "react";
import style from "./Home.module.css";
import Axios from "axios";

const Home = () => {

    const [checkStatus, setCheckStatus] = useState(false);
    const [urlData, setUrlData] = useState('');
    const [status, setStatus]  = useState('');

    const handleUserInput = event => {
        setUrlData(event.target.value);
    }

    const handleCheck = event => {

        Axios.post('http://localhost:8000/', {url: urlData}).then(response => setStatus(response.data.message));

        setUrlData('');
        setCheckStatus(!checkStatus);
    }

    return (
            <div className={style.container}>
                {!checkStatus ? (

                    <div className={style.table}>
                    <div className={style.input_wraper}>
                        <input className={style.user_input} type="text" placeholder="Enter URL" value={urlData} onChange={handleUserInput}></input>
                    </div>
                    <div className={style.input_wraper}>
                        <button className={style.user_input} type="submit" onClick={handleCheck}>CHECK</button>
                    </div>
                    </div>
                ):(
                    <div className={style.table}>
                        <h1>{status}  </h1>
                        <div className={style.input_wraper}>
                            <button className={style.user_input} onClick={handleCheck}>CHECK AGAIN</button>
                        </div>
                    </div>
                )}
            </div>
            )
}

export default Home;