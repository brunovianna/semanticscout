from numpy.random import default_rng
import torch
import math

class SemanticScout:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "cond": ("CONDITIONING", ),
                "radius": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "nsphere"
    OUTPUT_IS_LIST = (True, )
    #OUTPUT_NODE = False

    CATEGORY = "conditioning"

    global rng 
    rng = default_rng()

    def random_point_in_sphere(self,cond_length,radius):

        random_vector = [rng.standard_normal() for i in range(0,cond_length) ]
        inv_len = radius / math.sqrt(sum(coord * coord for coord in random_vector))
        return [coord * inv_len for coord in random_vector]

    def nsphere(self,cond,radius):
        print ("------------")
        print (len(cond))  
        print (len(cond[0]))
        print ("len cond[0][0] "+str(len(cond[0][0])))
        print ("len cond[0][1] "+str(len(cond[0][1])))
        print (cond[0][1].keys)
        t = torch.clone (cond[0][0])
        print (t.shape)
        #comfy.utils.save_torch_file(output, file, metadata=metadata)

        #even if I received multiple conditionings, I'll take just the first one and make num_points around it
        conds = []

        first_cond_tokens = torch.clone(cond[0][0][0])

        if "pooled_output" in cond[0][1]:
          first_cond_pooled = torch.clone(cond[0][1]["pooled_output"])
        

        # a vector of random nums the same lenght as the conditioning
        # cond data structure -> list_of_conds, normal+poolm, 1-element tensor, tokens, token_vector
        sphere_point = torch.FloatTensor(self.random_point_in_sphere(len(cond[0][0][0][0]),radius))
        print ("sphere point "+str(sphere_point[0]))
        #print ("0000 antes"+str(cond[0][0][0][0]))
        print ("len 000 "+str(len((cond[0][0][0]))))
        print ("len 0000 "+str(len((cond[0][0][0][0]))))

        print ("torch shape cond "+str((cond[0][0][0][0]).shape))
        print ("sphere shape "+str((sphere_point.shape)))

        for index in range (len(cond[0][0][0])):
            cond[0][0][0][index] = first_cond_tokens[index] + sphere_point

        d = cond[0][1].copy()
        if "pooled_output" in d:
            cond[0][1]["pooled_output"] = first_cond_pooled + sphere_point

        print ("result cond "+str((cond[0][0][0][0][0])))

        conds.append(cond)


        # print ("radius "+ str(radius))
        # print ("im in  testfunction")
        # print ("len "+str(len(cond[0])))
        # print ("shape 00 "+str(cond[0][0].shape))
        # print ("0000 depois"+str(cond[0][0][0][0]))
        # print ("0001 "+str(cond[0][0][0][1]))
        # print ("0002 "+str(cond[0][0][0][2]))
        # print ("shape 10 "+str(len(cond[0][1])))
        # print ("01 "+str(cond[0][1]["pooled_output"].shape))
        # print ("1001 "+str(cond[1][0][0][1]))
        # print ("1002 "+str(cond[1][0][0][2]))

        # c = []
        # for t in conditioning:
        #     d = t[1].copy()
        #     if "pooled_output" in d:
        #         d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        #     n = [torch.zeros_like(t[0]), d]
        #     c.append(n)


        print (len(conds))        
        return (conds,)
        
    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SemanticScout": SemanticScout
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticScout": "Semantic Scout"
}
