import React, { useState, useEffect } from "react";
import {
    Container,
    Heading,
    VStack,
    HStack,
    Flex,
    Textarea
} from "@chakra-ui/react";
import { Button, Upload, List } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const FileUploader = () => {
    const [fileList1, setFileList1] = useState([]);
    const [fileList2, setFileList2] = useState([]);
    const [logs, setLogs] = useState('');

    const handleUpload = async () => {
        const formData = new FormData();
        if (fileList1[0] && fileList2[0]) {
            formData.append("photozip", fileList1[0]);
            formData.append("targetphoto", fileList2[0]);

            setTimeout(() => {
                setLogs(prevLogs => `${prevLogs}\nAccept dataset and target image success`);
            }, 3000);

            setTimeout(() => {
                setLogs(prevLogs => `${prevLogs}\nStart generating verifier...`);
            }, 6000);

            setTimeout(async () => {
                setLogs(prevLogs => `${prevLogs}\nGenerate verifier success!`);
                const response = await fetch("http://127.0.0.1:5000/upload", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "Verifier.zip";
                    a.click();

                    const handleFocus = () => {
                        setLogs(prevLogs => `${prevLogs}\nDownload verifier success!`);
                        // 在日志更新后，移除事件监听器
                        window.removeEventListener('focus', handleFocus);
                    };

                    // 添加事件监听器
                    window.addEventListener('focus', handleFocus);

                } else {
                    console.error("Failed to download file");
                }
            }, 38000);

        }
    };

    const customRequest = ({ file, onSuccess }) => {
        setTimeout(() => {
            onSuccess("ok");
        }, 0);
        return file;
    };

    return (
        <Container
            minH="100vh"
            p={8}
            borderRadius="md"
            style={{
                backgroundImage: 'url(https://news.coincu.com/wp-content/uploads/2023/06/shutterstock_608453894.width-800.jpg)',
                backgroundSize: 'cover',
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'center',
            }}
        >
            <Flex
                width="100%"
                height="100%"
                direction="column"
                alignItems="center"
                justifyContent="center"
                paddingTop="25vh"
            >
                <VStack
                    spacing={20}
                    align="stretch"
                    p={8}
                    bg="transparent"
                    borderRadius="md"
                    boxShadow="md"
                    style={{ transform: "scale(1.5)" }}
                >
                    <Heading mb={4} color="white">ZKML Prover</Heading>
                    <HStack spacing={4}>
                        <Upload
                            customRequest={customRequest}
                            showUploadList={false}
                            beforeUpload={(file) => { setFileList1([file]); return false; }}
                        >
                            <Button icon={<UploadOutlined />}>Upload dataset.zip file</Button>
                        </Upload>
                        <Upload
                            customRequest={customRequest}
                            showUploadList={false}
                            beforeUpload={(file) => { setFileList2([file]); return false; }}
                        >
                            <Button icon={<UploadOutlined />}>Upload target image file</Button>
                        </Upload>
                    </HStack>
                    <List
                        size="small"
                        bordered
                        dataSource={fileList1.concat(fileList2)}
                        renderItem={file => <List.Item>{file.name}</List.Item>}
                    />
                    <Button
                        type="primary"
                        size="large"
                        onClick={handleUpload}
                    >
                        Download verifier
                    </Button>
                    <Textarea
                        value={logs}
                        isReadOnly
                        height="80px"
                        maxHeight="300px"
                    />
                </VStack>
            </Flex>
        </Container>
    );
};

export default FileUploader;
