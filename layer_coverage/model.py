import tensorflow as tf
import numpy as np

class Model:
    # tensorflow
    def __init__(self, model_path, framework):
        self.path = model_path
        self.framework = framework
        self.handler = Model.get_handler(self)
        self.inputs, self.outputs = Model.get_inputs_outputs(self)# tensor list

    @staticmethod
    def get_handler(self):
        if self.framework=='tensorflow':
            graph_def = tf.compat.v1.GraphDef()#tf.GraphDef()
            with open(self.path, "rb") as f:
                graph_def.ParseFromString(f.read())
            return graph_def
        elif self.framework=='tensorflowlite':
            interpreter = tf.lite.Interpreter(model_path=self.path)
            interpreter.allocate_tensors()
            return interpreter

    @staticmethod
    def get_inputs_outputs(self):
        if self.framework=='tensorflow':
            input_nodes = []
            variable_nodes = []
            output_nodes = []
            node2output = {}
            for i, n in enumerate(self.handler.node):
                # print(f'{i}-th node: {n.op} {n.name} {n.input}')
                if n.op == 'Placeholder':
                    input_nodes.append(n)
                if n.op in ['Variable', 'VariableV2']:
                    variable_nodes.append(n)
                for input_node in n.input:
                    node2output[input_node] = n.name
            for i, n in enumerate(self.handler.node):
                if n.name not in node2output and n.op not in ['Const', 'Assign', 'NoOp', 'Placeholder']:
                    output_nodes.append(n)
            if len(input_nodes) == 0 or len(output_nodes) == 0:
                return None
            return input_nodes, output_nodes
        elif self.framework=='tensorflowlite':
            return self.handler.get_input_details(), self.handler.get_output_details()
    
    def inference(self, inputs):
        if self.framework=='tensorflow':
            sess = tf.compat.v1.Session()
            tf.import_graph_def(self.handler, name='')  # import graph
            inputdic = {}
            outputlist = []
            for i in range(len(self.inputs)):
                input_tensor = sess.graph.get_tensor_by_name(self.inputs[i].name+':0')
                inputdic[input_tensor] = inputs[i]
            for o in range(len(self.outputs)):
                output_tensor = sess.graph.get_tensor_by_name(self.outputs[o].name+':0')
                outputlist.append(output_tensor)
            ret = sess.run(outputlist,feed_dict=inputdic) #[op1, op2]
            sess.close()
            return ret
        elif self.framework=='tensorflowlite':
            for i in range(len(self.inputs)):
                self.handler.set_tensor(self.inputs[i]['index'],inputs[i])
            self.handler.invoke()
            ret = []
            for o in range(len(self.outputs)):
                ret.append(self.handler.get_tensor(self.outputs[i]['index']))
            return ret
        return None
    
    def inference_single(self, _input):# single input single output
        outputnames = []
        if self.framework=='tensorflow':
            sess = tf.compat.v1.Session()
            tf.import_graph_def(self.handler, name='')  # import graph
            inputdic = {}
            outputlist = []
            input_tensor = sess.graph.get_tensor_by_name(self.inputs[0].name+':0')
            inputdic[input_tensor] = _input
            for i, n in enumerate(self.handler.node):
                if n.op.find('Conv2')!=-1 or n.op.find('Dense')!=-1 or n.op.find('MatMul')!=-1:
                    # print(n.name)
                    outputnames.append(n.name)
                    output_tensor = sess.graph.get_tensor_by_name(n.name+':0')
                    outputlist.append(output_tensor)
            # output_tensor = sess.graph.get_tensor_by_name(self.outputs[0].name+':0')
            # outputlist.append(output_tensor)
            # print(len(outputlist))
            ret = sess.run(outputlist,feed_dict=inputdic) #[op1, op2]
            sess.close()
            return ret, outputnames
        elif self.framework=='tensorflowlite':
            self.handler.set_tensor(self.inputs[0]['index'],_input)
            self.handler.invoke()
            # ret = []
            # ret = np.expand_dims(self.handler.get_tensor(self.outputs[0]['index']), axis=0)
            ret = self.handler.get_tensor(self.outputs[0]['index'])
            # ret = tf.convert_to_tensor(np.argmax(ret,-1))
            ret = tf.convert_to_tensor(ret)
            # import cv2
            # print(np.argmax(ret,-1))
            # array1=_input.numpy()
            # maxValue=array1.max()
            # array1=array1*255/maxValue
            # mat=np.uint8(array1)
            # cv2.imshow("test",mat[0])
            # cv2.waitKey(0)
            # ret.append(self.handler.get_tensor(self.outputs[0]['index']))
            return ret, outputnames
        return None, outputnames
 