import logo from './logo.svg';
import './App.css';
import FileUploader from './FileUploader';  // 假设 FileUploader.js 位于与 App.js 相同的目录中

function App() {
  // return (
  //   <div className="App">
  //     <header className="App-header">
  //       <img src={logo} className="App-logo" alt="logo" />
  //       <p>
  //         Edit <code>src/App.js</code> and save to reload.
  //       </p>
  //       <a
  //         className="App-link"
  //         href="https://reactjs.org"
  //         target="_blank"
  //         rel="noopener noreferrer"
  //       >
  //         Learn React
  //       </a>
  //     </header>
  //   </div>
  // );

  return (
      <div className="App">
        <FileUploader />
      </div>
  );
}

export default App;
